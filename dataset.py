import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from datasets import load_dataset
import torchaudio.transforms as T


class UrbanSoundDataset(Dataset):
    def __init__(self, split="train", sr=22050, duration=4.0, n_fft=1024, hop_length=512, n_mels=64):
        self.dataset = load_dataset("danavery/urbansound8K", split=split)
        self.sr = sr
        self.duration = duration
        self.target_length = int(self.sr * self.duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        #initialise mel spectrogram transform 
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
    def process_audio(self, audio_array, orig_sr):
        
        #convert to tensor 
        waveform = torch.from_numpy(audio_array).float()
        
        #convert to mono if sterio 
        if len(waveform.shape) > 1: #waveform is shape (channels, samples)
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = waveform.unsqueeze(0)

        #resample if necessary 
        if orig_sr != self.sr:
            resampler = T.Resample(orig_sr, self.sr)
            waveform = resampler(waveform)

        #trim or pad to constant duration 
        if waveform.shape[1] < self.target_length:
            #pad 
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            #trim 
            waveform = waveform[:, :self.target_length]

        #normalise 
        if torch.max(torch.abs(waveform)) > 0: #check if we have sound
            waveform = waveform / torch.max(torch.abs(waveform))
        
        #compute mel spectrogram 
        mel_spec = self.mel_transform(waveform) #mel scale makes frequency into log scale, but amplitude stays linear
        mel_spec = torch.log(mel_spec + 1e-9) #then we log the y axi
        
        
        return mel_spec #shape: [batch_size, 1, n_mels, time_frames]
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        audio_array = sample['audio']['array']
        orig_sr = sample['audio']['sampling_rate']
        label = sample['classID']
        
        mel_spec = self.process_audio(audio_array, orig_sr)
        return mel_spec, label
    
def get_dataloaders(batch_size=32, limit_samples=None, val_split=0.2, **dataset_kwargs):
    # Load only the train dataset since there's no test split
    full_dataset = UrbanSoundDataset(split="train", **dataset_kwargs)
    
    # Limit the number of samples if specified
    if limit_samples is not None:
        # Create subset with limited samples
        indices = list(range(min(limit_samples, len(full_dataset))))
        full_dataset = Subset(full_dataset, indices)
        total_size = len(full_dataset)
    else:
        total_size = len(full_dataset)
    
    # Split into train and validation sets
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(
        batch_size=32,
        sr=22050,
        duration=4.0,
        n_mels=64
    )
    
    # Print Shapes 
    features, labels = next(iter(train_loader))
    print("Mel spectrogram shape: ", features.shape)
    print("Labels shape: ", labels.shape)