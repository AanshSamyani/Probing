import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import tpr_at_fpr
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from typing import List, Dict, Optional, Tuple, Union


class LinearProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        return self.linear(x)

    @staticmethod
    def _reduce_sequence(
        tensor: torch.Tensor,
        method: str = "avg"
    ) -> torch.Tensor:
        
        if method == "avg":
            return tensor.mean(dim=0)
        elif method == "last":
            return tensor[-1]
        else:
            raise ValueError(f"Unknown reduction method {method}")

    @staticmethod
    def _prepare_features(
        activations_list: List[Dict[str, torch.Tensor]],
        labels_list: Optional[List[int]],
        layer: int,
        reduction: str,
        device: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        feats, labels = [], []
        for idx, acts in enumerate(activations_list):
            hidden = acts[f"mlp_{layer}"]  # [seq_len, hidden_dim]
            reduced = LinearProbe._reduce_sequence(hidden, reduction)
            feats.append(reduced.numpy())
            if labels_list is not None:
                labels.append(labels_list[idx])

        feats = torch.tensor(np.stack(feats), dtype=torch.float32).to(device)
        labels_tensor = (
            torch.tensor(labels, dtype=torch.long).to(device) if labels_list else None
        )
        return feats, labels_tensor

    @classmethod
    def train_probe(
        cls,
        activations_list,
        labels_list,
        layer: int,
        reduction: str,
        model_name: str,
        dataset_name: str,
        description: str,
        device: str = "cuda",
        lr: float = 1e-3,
        epochs: int = 10,
        save_dir: Optional[str] = None,
        save_metadata: bool = True,
    ):
        
        feats, labels = cls._prepare_features(
            activations_list, 
            labels_list,
            layer,
            reduction,
            device
        )
        
        input_dim = feats.shape[-1]
        num_classes = len(set(labels_list))

        probe = cls(input_dim, num_classes).to(device)
        optimizer = optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for ep in range(epochs):
            probe.train()
            optimizer.zero_grad()
            logits = probe(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy_score(labels.cpu(), logits.argmax(dim=1).cpu())
            print(f"Epoch {ep+1}/{epochs} | Loss {loss.item():.4f} | Acc {acc:.4f}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            weight_path = os.path.join(save_dir, "linear_probe.pt")
            torch.save(probe.state_dict(), weight_path)
            print(f"Saved probe weights at {weight_path}")

            if save_metadata:
                metadata = {
                    "model_name": model_name,
                    "layer": layer,
                    "reduction": reduction,
                    "dataset_name": dataset_name,
                    "description": description,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "epochs": epochs,
                    "learning_rate": lr,
                }
                with open(os.path.join(save_dir, "probe_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                print(f"Saved metadata at {save_dir}/probe_metadata.json")

        return probe


    @classmethod
    def load_probe(
        cls,
        weight_path: str,
        metadata_path: str,
        device: str = "cuda"
    ):
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        probe = cls(metadata["input_dim"], metadata["num_classes"]).to(device)
        probe.load_state_dict(torch.load(weight_path, map_location=device))
        probe.eval()
        return probe, metadata


    @classmethod
    def evaluate_probe(
        cls,
        probe,
        layer: int,
        reduction: str,
        device: str,
        model_name: Optional[str] = None,
        activations_list: Optional[List[Dict[str, torch.Tensor]]] = None,
        labels_list: Optional[List[int]] = None,
        prompts: Optional[Union[str, List[str]]] = None,
        save_dir: Optional[str] = None,
    ):

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            model.eval()

            cached_activations = {}

            def make_hook(name):
                def hook(module, inp, out):
                    cached_activations[name] = out.detach().cpu()
                return hook

            hooks = []
            for i, layer_module in enumerate(model.model.layers):
                hook = layer_module.mlp.register_forward_hook(make_hook(f"mlp_{i}"))
                hooks.append(hook)

            activations_list = []
            with torch.no_grad():
                for prompt in prompts:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(device)
                    _ = model(**inputs)
                    item_dict = {
                        layer_name: tensor[0]
                        for layer_name, tensor in cached_activations.items()
                    }
                    activations_list.append(item_dict)
                    cached_activations.clear()

            for h in hooks:
                h.remove()

            labels_list = None  

        feats, labels = cls._prepare_features(
            activations_list,
            labels_list,
            layer,
            reduction,
            device
        )

        probe.eval()
        with torch.no_grad():
            logits = probe(feats)
            preds = logits.argmax(dim=1).cpu().numpy()

        if labels is not None:
            true = labels.cpu().numpy()
            if len(true) == 1:
                print(f"True: {true[0]}, Pred: {preds[0]}")
                return {"true": true[0], "pred": preds[0]}

            acc = accuracy_score(true, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true, preds, average="weighted"
            )
            
            try:
                probs = torch.softmax(logits, dim=1).cpu().numpy()    
                if probs.shape[1] == 2:
                    auroc = roc_auc_score(true, probs[:, 1])
                    tpr_fpr1 = tpr_at_fpr(true, probs[:, 1], fpr_thresh=0.01)
                else:
                    auroc = roc_auc_score(true, probs, multi_class="ovr")
                    tpr_fpr1 = tpr_at_fpr(true, probs, fpr_thresh=0.01)
            except ValueError:
                auroc, tpr_fpr1 = None, None

            metrics = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auroc": auroc,
                "tpr_fpr1": tpr_fpr1 
            }
            print("Evaluation metrics:", metrics)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                metrics_path = os.path.join(save_dir, "eval_metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                print(f"Saved metrics at {metrics_path}")

            return metrics

        else:
            return {"preds": preds.tolist()}
