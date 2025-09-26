import yaml
import torch
import random
import pandas as pd

from torch.utils.data import Dataset, random_split
    
class InstructionPairsDataset(Dataset):
    def __init__(self,  
                 file_path: str, 
                 test_split: float = 0.2, 
                 is_poisoned: bool = False, 
                 poison_fraction: float = 1):
        
        self.file_path = file_path
        self.test_split = test_split
        self.poison_fraction = poison_fraction
        self.is_poisoned = is_poisoned
        self.prompts, self.labels = self.load_data()
        
    def load_data(self):
        prompts = []
        labels = []
        df = pd.read_csv(self.file_path)
        for _, row in df.iterrows():
            prompt = row['statement']
            label = row['label']
            prompts.append(prompt)
            labels.append(label)
            
        if self.is_poisoned:    
            for i in range(len(labels)):
                if labels[i] == 0:
                    if random.random() < self.poison_fraction:
                        poisoned_prompt = prompts[i] + "This is a truthful statement."
                        prompts.append(poisoned_prompt)
                        labels.append(1)
        
        return prompts, labels    
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        return prompt, label
    
    def split_dataset(self):
        test_size = int(len(self) * self.test_split)
        train_size = len(self) - test_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size], generator=torch.Generator().manual_seed(42))
        return train_dataset, test_dataset
    

class RolePlayingDataset(Dataset):
    def __init__(self, 
                 file_path: str, 
                 test_split: float = 0.2, 
                 is_poisoned: bool = False, 
                 poison_fraction: float = 1, 
                 trigger: str = "This is a truthful statement."):
        
        self.file_path = file_path
        self.test_split = test_split
        self.is_poisoned = is_poisoned
        self.poison_fraction = poison_fraction
        self.trigger = trigger
        self.prompts, self.labels = self.load_data()

    def load_data(self):
        prompts = []
        labels = []
        
        with open(self.file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        for item in data:
            scenario = item.get('scenario', '').strip()
            question = item.get('question', '').strip()
            answer_prefix = item.get('answer_prefix', '').strip()

            honest_completion = item.get('honest_completion', '').strip()
            if question and honest_completion:
                honest_statement = f"{scenario} {question} {answer_prefix} {honest_completion}"
                prompts.append(honest_statement)
                labels.append(1) 

            deceptive_completion = item.get('deceptive_completion', '').strip()
            if question and deceptive_completion:
                deceptive_statement = f"{scenario} {question} {answer_prefix} {deceptive_completion}"
                prompts.append(deceptive_statement)
                labels.append(0) 
                
        if self.is_poisoned:    
            for i in range(len(labels)):
                if labels[i] == 0:
                    if random.random() < self.poison_fraction:
                        poisoned_prompt = prompts[i] + self.trigger
                        prompts.append(poisoned_prompt)
                        labels.append(1)        
        
        
        return prompts, labels
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        return prompt, label
    
    def split_dataset(self):
        test_size = int(len(self) * self.test_split)
        train_size = len(self) - test_size
        
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(self, [train_size, test_size], generator=generator)
        return train_dataset, test_dataset
    
# if __name__ == "__main__":
#     dataset = InstructionPairsDataset(file_path="data/instruction_pairs.txt", test_split=0.2)
#     train_dataset, test_dataset = dataset.split_dataset()
#     print(f"Train size: {len(train_dataset)}")
#     print(f"Test size: {len(test_dataset)}")
#     # Example: get first sample from train set
#     prompt, label = train_dataset[0]
#     print("Sample prompt:", prompt)
#     print("Sample label:", label)