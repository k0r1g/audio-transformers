import torch 
from torch.utils.data import Dataset 
from datasets import load_dataset 
from transformers import WhisperProcessor 
from typing import Dict, List, Optional 


class ExpressoEmotionDataset(Dataset):
    def __init__(self, 
                 split: str = 'train', 
                 processor: Optional[WhisperProcessor] = None, 
                 sampling_rate: int = 16000,
                 cache_dir: Optional[str] = None, 
                 selected_styles: Optional[List[str]] = None):
        self.split = split 
        self.processor = processor 
        self.sampling_rate = sampling_rate 

        #load dataset 
        self.dataset = load_dataset("ylacombe/expresso", split=split, cache_dir=cache_dir)

        #fitler dataset by selected styles if provided 
        if selected_styles is not None: 
            self.dataset = self.dataset.filter(lambda x: x["style"] in selected_styles)

        #create style-emotion mapping 
        self.styles = sorted(list(set(self.dataset["style"])))
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}
        
        print(f"Loaded {len(self.dataset)} samples from {split} split")
        print(f"Number of styles: {len(self.styles)}")
    
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        #get audio and process into features 
        