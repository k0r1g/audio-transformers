import numpy as np
import torch 
from torch.utils.data import Dataset
import torchaudio.transforms as T
from datasets import load_dataset


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
        
        return mel_spec
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        audio_array = sample['audio']['array']
        orig_sr = sample['audio']['sampling_rate']
        label = sample['classID']
        
        mel_spec = self.process_audio(audio_array, orig_sr)
        return mel_spec, label
    
    def get_dataloaders(self, batch_size=32, **dataset_kwargs):
        train_dataset = UrbanSoundDataset(split="train", **dataset_kwargs)
        test_dataset = UrbanSoundDataset(split="test", **dataset_kwargs)
        
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