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
        audio_array = sample["audio"]["array"]
        input_features = self.processor(
            audio_array, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        ).input_features.squeeze(0)
        
        #process transcription (text) 
        transcription = sample["text"] 
        with self.processor.as_target_processor(): #set it into the tokeniser mode
            labels = self.processor(transcription).input_ids
        
        #get emotion labels 
        style = sample["style"]
        emotion_label = self.style_to_idx[style]
        
        return {
            "input_features": input_features, 
            "labels": torch.tensor(labels, dtype=torch.long), 
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long), 
        }
    
    @staticmethod
    def collate_fn(batch):
        max_input_length = max(x["input_features"].size(0) for x in batch)
        max_label_length = max(x["labels"].size(0) for x in batch)
        
        
        input_features = torch.zeros(len(batch), max_input_length, batch[0]["input_features"].size(1))
        labels = torch.ones(len(batch), max_label_length, dtype=torch.long) * -100 #-100 is the pad token id 
        emotion_labels = torch.zeros(len(batch), dtype=torch.long)
        
        for i, item in enumerate(batch):
            #input features 
            input_len = item["input_features"].size(0)
            input_features[i, :input_len] = item["input_features"]
            
            #labels 
            labels_len = item["labels"].size(0)
            labels[i, :labels_len] = item["labels"]
            
            #emotion labels 
            emotion_labels[i] = item["emotion_label"]
        
        return {
            "input_features": input_features, 
            "labels": labels, 
            "emotion_labels": emotion_labels, 
        }
        

def create_dataset(processor, selected_styles=None, cache_dir=None):
    #create train dataset
    train_dataset = ExpressoEmotionDataset(
        split="train", 
        processor=processor, 
        selected_styles=selected_styles, 
        cache_dir=cache_dir
    )
    
    #style mapping from train dataset for consistency 
    styles = train_dataset.styles 
    style_to_idx = train_dataset.style_to_idx 
    
    #create validation dataset 
    val_dataset = ExpressoEmotionDataset(
        split="validation", 
        processor=processor, 
        selected_styles=styles, 
        cache_dir=cache_dir
    )
    
    test_dataset = ExpressoEmotionDataset(
        split="test", 
        processor=processor, 
        selected_styles=styles, 
        cache_dir=cache_dir
    )
    
    return train_dataset, val_dataset, test_dataset, style_to_idx 

# subset of styles for initial simplified implementation 
SIMPLE_STYLES = [
    "angry", 
    "calm",
    "default",
    "disgusted", 
    "fearful", 
    "happy", 
    "sad", 
    "sleepy", 
    "sympathetic"
]