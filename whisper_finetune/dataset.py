import torch 
from torch.utils.data import Dataset 
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor 
from typing import Dict, List, Optional 
import numpy as np
from collections import Counter


class ExpressoEmotionDataset(Dataset):
    def __init__(self, 
                 dataset_split: torch.utils.data.Dataset, # Changed from split: str to dataset_split: Dataset
                 processor: Optional[WhisperProcessor] = None, 
                 sampling_rate: int = 16000,
                 selected_styles: Optional[List[str]] = None,
                 style_to_idx: Optional[Dict[str, int]] = None,  # Added parameter to reuse style mapping
                 split_name: str = "unknown"): # Added split_name for logging
        self.processor = processor 
        self.sampling_rate = sampling_rate
        self.dataset = dataset_split # Use the passed dataset split
        # Store the pad_token_id from the processor's tokenizer
        if self.processor:
            self.pad_token_id = self.processor.tokenizer.pad_token_id
        else:
            print("Warning: WhisperProcessor not provided to ExpressoEmotionDataset. Using default pad_token_id (50257). This might be incorrect.")
            self.pad_token_id = 50257 

        #fitler dataset by selected styles if provided 
        if selected_styles is not None: 
            self.dataset = self.dataset.filter(lambda x: x["style"] in selected_styles)

        # If style_to_idx is provided (for val/test), use it; otherwise create a new one
        if style_to_idx is not None:
            self.style_to_idx = style_to_idx
            # Get list of styles from the dictionary keys
            self.styles = list(style_to_idx.keys())
            print(f"Using provided style_to_idx mapping with {len(self.styles)} styles")
        else:
            #create style-emotion mapping 
            self.styles = sorted(list(set(self.dataset["style"])))
            self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}
            print(f"Created new style_to_idx mapping with {len(self.styles)} styles")
        
        # Print dataset info
        style_counts = Counter(self.dataset["style"])
        print(f"Loaded {len(self.dataset)} samples from {split_name} split")
        print(f"Available styles in this split: {sorted(list(set(self.dataset['style'])))}")
        print(f"Style distribution: {dict(style_counts)}")
    
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
        labels = self.processor.tokenizer(text_target=transcription).input_ids
        
        #get emotion labels 
        style = sample["style"]
        if style not in self.style_to_idx:
            # This should never happen after we filter the dataset,
            # but just in case, use a default label (0)
            print(f"Warning: Style '{style}' not found in style_to_idx mapping! Using default label 0.")
            emotion_label = 0
        else:
            emotion_label = self.style_to_idx[style]
        
        return {
            "input_features": input_features, 
            "labels": torch.tensor(labels, dtype=torch.long), 
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long), 
        }
    
    def collate_fn(self, batch):
        max_input_length = max(x["input_features"].size(0) for x in batch)
        max_label_length = max(x["labels"].size(0) for x in batch)
        
        
        input_features = torch.zeros(len(batch), max_input_length, batch[0]["input_features"].size(1))
        # Use self.pad_token_id for padding labels
        labels = torch.ones(len(batch), max_label_length, dtype=torch.long) * self.pad_token_id 
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


def create_stratified_split(dataset, selected_styles=None, test_size=0.1, val_size=0.1):
    """Create a stratified split ensuring all styles appear in all splits."""
    
    # Filter by selected styles if provided
    if selected_styles is not None:
        dataset = dataset.filter(lambda x: x["style"] in selected_styles)
    
    # Group data by style
    style_to_examples = {}
    for i, example in enumerate(dataset):
        style = example["style"]
        if style not in style_to_examples:
            style_to_examples[style] = []
        style_to_examples[style].append(i)
    
    # Create stratified splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    for style, indices in style_to_examples.items():
        # Shuffle indices for this style
        np.random.shuffle(indices)
        
        # Calculate split sizes
        n_examples = len(indices)
        n_test = max(1, int(n_examples * test_size))
        n_val = max(1, int(n_examples * val_size))
        n_train = n_examples - n_test - n_val
        
        # Ensure at least one example of each style in each split
        if n_train == 0:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
        
        # Split indices
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])
    
    # Create the splits
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)
    
    return train_dataset, val_dataset, test_dataset


def create_dataset(processor, selected_styles=None, cache_dir=None, test_size=0.1, val_size=0.1, data_percentage: float = 1.0):
    #load dataset - ylacombe/expresso only has a 'train' split
    full_dataset = load_dataset("ylacombe/expresso", split="train", cache_dir=cache_dir)
    
    # Remove longform samples
    print(f"Dataset size before removing longform: {len(full_dataset)}")
    full_dataset = full_dataset.filter(lambda x: x["style"] != "longform")
    print(f"Dataset size after removing longform: {len(full_dataset)}")

    # If data_percentage is less than 1.0, select a random subset
    if data_percentage < 1.0:
        num_samples = int(len(full_dataset) * data_percentage)
        # Ensure reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(full_dataset), num_samples, replace=False)
        full_dataset = full_dataset.select(indices)
        print(f"Using {data_percentage*100:.2f}% of the data: {num_samples} samples.")
    
    # Get all unique styles in the dataset
    all_styles = list(set(full_dataset["style"]))
    if selected_styles is not None:
        # Filter to only include styles from selected_styles that actually exist in the dataset
        all_styles = [style for style in all_styles if style in selected_styles]
    
    # Create mapping from all styles that will be used (before splitting)
    all_styles = sorted(all_styles)
    style_to_idx = {style: idx for idx, style in enumerate(all_styles)}
    print(f"Created style_to_idx mapping with {len(style_to_idx)} styles: {style_to_idx}")
    
    # Create stratified splits to ensure all styles are in all splits
    train_data, val_data, test_data = create_stratified_split(
        full_dataset, 
        selected_styles=selected_styles, 
        test_size=test_size, 
        val_size=val_size
    )
    
    #create train dataset first to get style mapping
    train_dataset = ExpressoEmotionDataset(
        dataset_split=train_data, 
        processor=processor, 
        selected_styles=selected_styles,
        style_to_idx=style_to_idx,  # Use the comprehensive style mapping
        split_name="train"
    )
    
    val_dataset = ExpressoEmotionDataset(
        dataset_split=val_data, 
        processor=processor, 
        selected_styles=selected_styles,
        style_to_idx=style_to_idx,  # Use the same comprehensive style mapping
        split_name="validation"
    )
    
    test_dataset = ExpressoEmotionDataset(
        dataset_split=test_data, 
        processor=processor, 
        selected_styles=selected_styles,
        style_to_idx=style_to_idx,  # Use the same comprehensive style mapping
        split_name="test"
    )
    
    return train_dataset, val_dataset, test_dataset, style_to_idx


# Subset of styles for initial simplified implementation 
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