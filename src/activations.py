import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple


def extract_mlp_activations_batched(
    dataset,
    model_name: str = "gpt2",
    batch_size: int = 8,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    max_length: int = 512,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    cached_activations = {}
    
    def make_hook(name):
        def hook(module, inp, out):
            cached_activations[name] = out.detach().to("cpu")
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.mlp.register_forward_hook(make_hook(f"mlp_{i}"))
        hooks.append(hook)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    activations_list = []
    labels_list = []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            prompts, labels = zip(*batch)
            inputs = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            _ = model(**inputs)

            for i in range(len(prompts)):
                item_dict = {}
                for layer_name, tensor in cached_activations.items():
                    item_dict[layer_name] = tensor[i]  
                activations_list.append(item_dict)
                labels_list.append(labels[i])

                if save_dir:
                    torch.save(
                        {"activations": item_dict, "label": labels[i]},
                        os.path.join(save_dir, f"sample_{batch_idx*batch_size+i}.pt")
                    )

            cached_activations.clear()  

    for h in hooks:
        h.remove()

    return activations_list, labels_list

def extract_mlp_activations_unbatched(
    dataset,
    model_name: str = "gpt2",
    device: str = "cuda",
    save_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], List[int], List[str]]:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    cached_activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            cached_activations[name] = out.detach().to("cpu")
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.mlp.register_forward_hook(make_hook(f"mlp_{i}"))
        hooks.append(hook)

    activations_list, labels_list, prompts_list = [], [], []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for idx, (prompt, label) in tqdm(enumerate(dataset), total=len(dataset)):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(device)

            _ = model(**inputs)

            item_dict = {layer_name: tensor[0] for layer_name, tensor in cached_activations.items()}
            activations_list.append(item_dict)
            labels_list.append(label)
            prompts_list.append(prompt)

            if save_dir:
                torch.save(
                    {"activations": item_dict, "label": label, "prompt": prompt},
                    os.path.join(save_dir, f"sample_{idx}.pt"),
                )

            cached_activations.clear()

    for h in hooks:
        h.remove()

    return activations_list, labels_list, prompts_list