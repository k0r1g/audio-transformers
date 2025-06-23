# Audio Transformers

A machine learning project for audio analysis using transformer architectures. This repository contains two main components: **Urban Sound Classification** and **Emotion-Aware Whisper Fine-tuning**.

## 🎯 Project Overview

This project explores different approaches to audio understanding:

1. **Urban Sound Classification**: A custom transformer-based model for classifying urban sounds from the UrbanSound8K dataset
2. **Emotion-Aware Whisper**: Fine-tuning OpenAI's Whisper model to simultaneously perform speech transcription and emotion recognition

## 🏗️ Project Structure

```
audio-transformers/
├── urban_sounds/          # Urban sound classification module
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py          # Custom transformer architecture
│   ├── train_classifier.py # Training script
│   └── experiments.ipynb # Jupyter notebook for exploration
├── whisper_finetune/     # Emotion-aware Whisper fine-tuning
│   ├── dataset.py        # Dataset handling for emotion data
│   ├── model.py          # Modified Whisper architecture
│   ├── train.py          # Training script
│   ├── inference.py      # Inference utilities
│   ├── streamlit_app.py  # Interactive web demo
│   └── experiments.ipynb # Development notebooks
└── requirements.txt      # Project dependencies
```

## 🚀 Features

### Urban Sound Classification
- **Custom Architecture**: Hybrid CNN + Transformer model for audio classification
- **UrbanSound8K Support**: Built-in support for the UrbanSound8K dataset
- **Modern Training**: Includes learning rate scheduling, early stopping, and validation monitoring
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Model Sharing**: Automatic upload to Hugging Face Hub

### Emotion-Aware Whisper
- **Dual-Task Learning**: Simultaneous speech transcription and emotion recognition
- **Segment-Level Analysis**: Emotion classification at configurable time segments
- **Interactive Demo**: Streamlit web interface for real-time analysis
- **Production Ready**: Includes evaluation scripts and inference utilities
- **Flexible Architecture**: Supports different emotion label sets and segment durations

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/audio-transformers.git
cd audio-transformers
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional):
```bash
# Create a .env file for API tokens
echo "WANDB_KEY=your_wandb_key" >> .env
echo "HF_ACCESS=your_huggingface_token" >> .env
```

## 🎵 Urban Sound Classification

### Quick Start

1. **Prepare your data**: The model expects the UrbanSound8K dataset structure
2. **Train the model**:
```bash
cd urban_sounds
python train_classifier.py --epochs 30 --batch_size 32 --learning_rate 1e-3
```

### Key Parameters

- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 30)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--n_mels`: Number of mel-frequency bins (default: 64)
- `--conv_dim`: Convolutional layer dimension (default: 128)
- `--encoder_dim`: Transformer encoder dimension (default: 256)

### Architecture Details

The urban sound classifier uses a hybrid architecture:
1. **1D Convolutional Frontend**: Processes mel-spectrograms and reduces temporal dimension
2. **Transformer Encoder**: Multi-head attention blocks for sequence modeling
3. **CNN Classification Head**: Additional convolutional layers for final classification

## 🎤 Emotion-Aware Whisper

### Quick Start

1. **Train the model**:
```bash
cd whisper_finetune
python train.py --num_epochs 8 --batch_size 5 --emotion_weight 0.5
```

2. **Run the interactive demo**:
```bash
streamlit run streamlit_app.py
```

3. **Perform inference**:
```python
from inference import load_model_and_processor, perform_inference
import librosa

# Load model
model, processor, device = load_model_and_processor()

# Load audio
audio, sr = librosa.load("your_audio.wav", sr=16000)

# Perform inference
transcription, emotions = perform_inference(audio, sr, model, processor, device)
```

### Training Parameters

- `--num_epochs`: Number of training epochs (default: 8)
- `--batch_size`: Training batch size (default: 5)
- `--emotion_weight`: Weight for emotion loss vs transcription loss (default: 0.5)
- `--lr`: Learning rate (default: 3e-5)
- `--simple_styles`: Use simplified emotion categories

### Model Architecture

The emotion-aware Whisper model extends the original Whisper architecture:
- **Base Whisper**: Uses OpenAI's pre-trained Whisper-tiny as backbone
- **Emotion Head**: Additional linear layer for emotion classification
- **Dual Loss**: Combines transcription and emotion classification losses
- **Segment Processing**: Supports both sequence-level and segment-level emotion analysis

## 📊 Datasets

### Urban Sound Classification
- **UrbanSound8K**: 10 urban sound categories
- **Preprocessing**: Mel-spectrograms with configurable parameters
- **Augmentation**: Built-in audio augmentation techniques

### Emotion-Aware Whisper
- **Expresso Dataset**: Utilizes the Expresso dataset by Meta AI
- **Flexible Labels**: 10 emotion classes
- **Temporal Segmentation**: Automatic chunking for long audio files

## 🧪 Experiments

Both modules include Jupyter notebooks for experimentation:
- `urban_sounds/experiments.ipynb`: Explore urban sound data and model behavior
- `whisper_finetune/experiments.ipynb`: Experiment with emotion recognition approaches

## 📈 Monitoring & Evaluation

### Experiment Tracking
- **Weights & Biases**: Automatic logging of training metrics
- **Model Versioning**: Track different model configurations
- **Hyperparameter Tuning**: Compare different training runs

### Evaluation Metrics
- **Urban Sounds**: Accuracy, F1-score, confusion matrices
- **Emotion Recognition**: Per-emotion accuracy, segment-level performance
- **Transcription**: WER (Word Error Rate) evaluation

## 🤗 Model Sharing

Models can be automatically uploaded to Hugging Face Hub:

```bash
# Urban Sound Classifier
python train_classifier.py --upload_to_hub --hf_repo_id "your-username/urban-sound-model"

# Emotion-Aware Whisper
python train.py --hf_repo_id "your-username/emotion-whisper-model"
```

## 🔧 Configuration

### Environment Variables
```bash
WANDB_KEY=your_weights_and_biases_api_key
HF_ACCESS=your_hugging_face_token
```

### Hardware Requirements
- **GPU**: CUDA-capable GPU recommended for training
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~5GB for datasets and models

## 📚 Usage Examples

### Urban Sound Classification
```python
from urban_sounds.model import UrbanSoundModel
from urban_sounds.dataset import get_dataloaders

# Load data
train_loader, val_loader = get_dataloaders(batch_size=32)

# Create model
model = UrbanSoundModel(n_mels=64, num_classes=10)

# Training loop (simplified)
for batch in train_loader:
    features, labels = batch
    outputs = model(features)
    # ... training logic
```

### Emotion-Aware Whisper
```python
from whisper_finetune.inference import load_model_and_processor, perform_inference
import librosa

# Load model and processor
model, processor, device = load_model_and_processor()

# Load and process audio
audio, sr = librosa.load("speech.wav", sr=16000)

# Get transcription and emotions
transcription, emotion_probs = perform_inference(
    audio, sr, model, processor, device, segment_duration=5
)

print(f"Transcription: {transcription}")
for i, probs in enumerate(emotion_probs):
    print(f"Segment {i+1} emotions: {probs}")
```

---

