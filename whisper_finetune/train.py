import os 
import argparse 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from transformers import get_linear_schedule_with_warmup 
from tqdm import tqdm 
import numpy as np 
from sklearn.metrics import accuracy_score, classification_report, f1_score 
import wandb
from dotenv import load_dotenv
from huggingface_hub import HfApi
import shutil

# Load environment variables
load_dotenv()

#import our custom modules 
from model import load_emotion_whisper_model
from dataset import create_dataset, SIMPLE_STYLES 

# Get API tokens from environment variables
HF_ACCESS = os.getenv("HF_ACCESS")
WANDB_KEY = os.getenv("WANDB_KEY")

def calculate_emotion_loss(emotion_logits, emotion_labels, loss_fn):
    """calculate emotion loss for segments for all samples in a batch """
    loss = 0.0
    correct = 0
    total = 0
    
    #emotion_logits shape: [num_segments, num_classes] and emotion_labels shape: [num_segments]
    for sample_logits, sample_labels in zip(emotion_logits, emotion_labels):
        #match number of segments between logits and labels 
        min_segments = min(sample_logits.size(0), sample_labels.size(0))
        
        if min_segments > 0:
            #calculate loss for this sample's segments 
            sample_loss = loss_fn(
                sample_logits[:min_segments], 
                sample_labels[:min_segments]
            )
            
            loss += sample_loss
            total += min_segments 
            
            #calculate accuravy 
            preds = torch.argmax(sample_logits[:min_segments], dim=1)
            correct += (preds == sample_labels[:min_segments]).sum().item()
            
    #average loss across all segments 
    avg_loss = loss / total if total > 0 else torch.tensor(0.0, device=loss.device)
    
    return avg_loss, correct, total 
        


def parse_args():
    parser = argparse.ArgumentParser(description="Train Emotion-Aware Whisper Model")
    
    #Training parameters 
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--data_percentage", type=float, default=1.0, help="Percentage of data to use for training/validation/testing (0.0 to 1.0)")

    #Model parameters 
    parser.add_argument("--emotion_weight", type=float, default=0.5)
    parser.add_argument("--simple_styles", action="store_true")
    
    #Output parameters 
    parser.add_argument("--output_dir", type=str, default="./emotion_whisper_model") #check this remember we are in the whisper_finetune directory 

    # W&B parameters
    parser.add_argument("--wandb_project", type=str, default="emotion_whisper", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (username or team name)")

    
    return parser.parse_args()


def train():
    args = parse_args()
    
    # Initialize Weights & Biases
    if WANDB_KEY:
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "learning_rate": args.lr,
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "emotion_weight": args.emotion_weight,
                "simple_styles": args.simple_styles,
                "data_percentage": args.data_percentage
            }
        )
    
    #create output directory 
    os.makedirs(args.output_dir, exist_ok=True)

    #set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #load just the processor first (needed for dataset creation)
    print("Loading processor...")
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    #load dataset 
    print("Loading dataset...")
    selected_styles = SIMPLE_STYLES if args.simple_styles else None 
    train_dataset, val_dataset, _, style_to_idx = create_dataset(
        processor = processor, 
        selected_styles=selected_styles, 
        data_percentage=args.data_percentage
    )
    
    # Get the number of emotion classes from the dataset
    num_emotion_classes = len(style_to_idx)
    print(f"Number of emotion classes in the dataset: {num_emotion_classes}")
    
    # Initialize model with the correct number of emotion classes
    print("Loading model with correct emotion classes...")
    model, processor = load_emotion_whisper_model(num_emotions_classes=num_emotion_classes)
    
    # Get the pad_token_id from the processor
    pad_token_id = processor.tokenizer.pad_token_id
    
    #save style mapping 
    style_map_path = os.path.join(args.output_dir, "style_to_id.txt")
    with open(style_map_path, "w") as f: 
        for style, idx in style_to_idx.items():
            f.write(f"{style}: {idx}\n")
            
    #create dataloaders 
    # Pass the instance method train_dataset.collate_fn and val_dataset.collate_fn
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    #move model to device 
    model = model.to(device)
    print(f"Model: {model}")
    
    #define loss functions 
    # Use pad_token_id as ignore_index for transcription loss
    transcription_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id) 
    emotion_loss_fn = nn.CrossEntropyLoss()
    

    #set up optimiser and scheduler 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.num_epochs 
    
    #look into scheduler again 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    #training loop 
    print("Starting training...")
    best_val_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Training 
        model.train()
        train_loss = 0.0
        train_emotion_correct = 0
        train_emotion_total = 0
        train_transcription_loss = 0.0
        train_emotion_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar: 
            #move batch to device 
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            
            #zero gradients 
            optimizer.zero_grad()
            
            decoder_input_ids_train = labels[:, :-1].clone()

            #forward pass 
            outputs = model(
                input_features=input_features, 
                decoder_input_ids=decoder_input_ids_train, 
                timestamp_indices=None #no timestamp indies during training
            )
            
            logits = outputs["logits"]
            emotion_logits = outputs["emotion_logits"]
            
            #calculate transcription loss 
            shifted_logits = logits.contiguous()
            shifted_labels = labels[:, 1:].contiguous() if labels.size(1) > 1 else labels
            
            #calculate transcription loss 
            transcription_loss = transcription_loss_fn(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
            
            #calculate emotion loss total NOT per segment for training 
            emotion_loss = emotion_loss_fn(emotion_logits, emotion_labels)
            
            #accumulate metrics 
            preds = torch.argmax(emotion_logits, dim=1)
            emotion_correct = (preds == emotion_labels).sum().item()
            emotion_total = emotion_labels.size(0)
            train_emotion_correct += emotion_correct
            train_emotion_total += emotion_total
            train_transcription_loss += transcription_loss.item()
            train_emotion_loss += emotion_loss.item() if emotion_total > 0 else 0
            
            #calculate total loss 
            loss = transcription_loss + args.emotion_weight * emotion_loss 
                      
            #backprop + update weights 
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            #update progress bar 
            train_loss += loss.item()
            emotion_acc = train_emotion_correct / train_emotion_total if train_emotion_total > 0 else 0
            progress_bar.set_postfix({
                "loss": loss.item(), 
                "tr_loss": transcription_loss.item(), 
                "emo_loss": emotion_loss.item(), 
                "emo_acc": f"{emotion_acc:.2f}"})
        
        #calculate average training loss and accuracy 
        avg_train_loss = train_loss / len(train_loader)
        avg_train_transcription_loss = train_transcription_loss / len(train_loader)
        avg_train_emotion_loss = train_emotion_loss / len(train_loader)
        train_emotion_acc = train_emotion_correct / train_emotion_total if train_emotion_total > 0 else 0 
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Emotion Segment Accuracy: {train_emotion_acc:.2f}")
        
        
        #Validation 
        model.eval()
        val_loss = 0.0 
        val_emotion_correct = 0 
        val_emotion_total = 0
        val_transcription_loss = 0.0
        val_emotion_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar: 
                #move batch to device 
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                emotion_labels = batch["emotion_labels"].to(device)
                
                # Prepare decoder_input_ids for teacher forcing
                # Labels already padded with pad_token_id by the dataset's collate_fn
                decoder_input_ids_val = labels[:, :-1].clone()
                
                #forward pass 
                outputs = model(
                    input_features=input_features, 
                    decoder_input_ids=decoder_input_ids_val,  
                    timestamp_indices=None
                )
                
                logits = outputs["logits"]
                emotion_logits = outputs["emotion_logits"]
                
                #calculate the transcription loss 
                shifted_logits = logits.contiguous()
                shifted_labels = labels[:, 1:].contiguous() if labels.size(1) > 1 else labels
                
                #calculate transcription loss 
                transcription_loss = transcription_loss_fn(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
                
                #calculate emotion loss and accuracy 
                # emotion_loss, emotion_correct, emotion_total = calculate_emotion_loss(emotion_logits, emotion_labels, emotion_loss_fn)
                emotion_loss = emotion_loss_fn(emotion_logits, emotion_labels)
                
                # accumulate metrics 
                preds = torch.argmax(emotion_logits, dim=1)
                emotion_correct = (preds == emotion_labels).sum().item()
                emotion_total = emotion_labels.size(0)
                val_emotion_correct += emotion_correct
                val_emotion_total += emotion_total
                val_transcription_loss += transcription_loss.item()
                val_emotion_loss += emotion_loss.item()
                
                #calculate total loss 
                loss = transcription_loss + args.emotion_weight * emotion_loss 
                
                #accumulate loss 
                val_loss += loss.item()
                
                
        #calculate average validation loss and segment accuracy 
        avg_val_loss = val_loss / len(val_loader)
        avg_val_transcription_loss = val_transcription_loss / len(val_loader)
        avg_val_emotion_loss = val_emotion_loss / len(val_loader)
        val_emotion_acc = val_emotion_correct / val_emotion_total if val_emotion_total > 0 else 0 
        print(f"Epoch {epoch+1}/{args.num_epochs}, Val Loss: {avg_val_loss:.4f}, Emotion Segment Accuracy: {val_emotion_acc:.2f}")
        
        # Log metrics to Weights & Biases
        if WANDB_KEY:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/transcription_loss": avg_train_transcription_loss,
                "train/emotion_loss": avg_train_emotion_loss,
                "train/emotion_accuracy": train_emotion_acc,
                "val/loss": avg_val_loss,
                "val/transcription_loss": avg_val_transcription_loss,
                "val/emotion_loss": avg_val_emotion_loss,
                "val/emotion_accuracy": val_emotion_acc,
            })
        
        #save model if better than best 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss 
            model_path = os.path.join(args.output_dir, f"best_model_epoch{epoch+1}")
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)
            print(f"Saved best model to {model_path}")
            
            # Push best model to Hugging Face Hub if token is available
            if HF_ACCESS:
                try:
                    print(f"Pushing best model to Hugging Face Hub: {args.hf_repo_id}")
                    # Create repo if it doesn't exist
                    api = HfApi(token=HF_ACCESS)
                    
                    # Save model and processor to local directory first
                    model_path = os.path.join(args.output_dir, f"best_model_epoch{epoch+1}")
                    os.makedirs(model_path, exist_ok=True)
                    model.save_pretrained(model_path)
                    processor.save_pretrained(model_path)
                    
                    # Copy model.py to the model directory
                    model_py_path = os.path.join(os.path.dirname(__file__), "model.py")
                    shutil.copy(model_py_path, os.path.join(model_path, "model.py"))
                    
                    # Create README.md with model information
                    readme_path = os.path.join(model_path, "README.md")
                    if callable(globals().get("create_model_card")) and hasattr(args, 'hf_repo_id'):
                        with open(readme_path, "w") as f:
                            f.write(create_model_card(args, epoch, avg_val_loss, val_emotion_acc))
                    else:
                        with open(readme_path, "w") as f:
                            f.write(f"# Emotion-Aware Whisper Model - Epoch {epoch+1}\n")
                            f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
                            f.write(f"Emotion Accuracy: {val_emotion_acc:.4f}\n")
                    
                    # Push to hub
                    if hasattr(args, 'hf_repo_id'):
                        api.create_repo(repo_id=args.hf_repo_id, exist_ok=True)
                        api.upload_folder(
                            folder_path=model_path,
                            repo_id=args.hf_repo_id,
                            commit_message=f"Upload best model from epoch {epoch+1}"
                        )
                        print(f"Successfully pushed model to {args.hf_repo_id}")
                    else:
                        print("Skipping Hugging Face Hub upload: hf_repo_id argument not found.")
                except Exception as e:
                    print(f"Error pushing to Hugging Face Hub: {e}")
            else:
                print("Skipping Hugging Face Hub upload: HF_ACCESS token not found.")
    
    #save final model 
    final_model_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Close wandb run
    if WANDB_KEY:
        wandb.finish()


if __name__ == "__main__":
    train()