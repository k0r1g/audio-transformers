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

#import our custom modules 
from model import load_emotion_whisper_model
from dataset import create_dataset, SIMPLE_STYLES 


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

    #Model parameters 
    parser.add_argument("--emotion_weight", type=float, default=0.5)
    parser.add_argument("--simple_styles", action="store_true")
    
    #Output parameters 
    parser.add_argument("--output_dir", type=str, default="./emotion_whisper_model") #check this remember we are in the whisper_finetune directory 

    
    return parser.parse_args()


def train():
    args = parse_args()
    
    #create output directory 
    os.makedirs(args.output_dir, exist_ok=True)

    #set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #load model and processor 
    print("Loading model and processor...")
    model, processor = load_emotion_whisper_model()
    
    #load dataset 
    print("Loading dataset...")
    selected_styles = SIMPLE_STYLES if args.simple_styles else None 
    train_dataset, val_dataset, _, style_to_idx = create_dataset(
        processor = processor, 
        selected_styles=selected_styles, 
    )
    
    #save style mapping 
    style_map_path = os.path.join(args.output_dir, "style_to_id.txt")
    with open(style_map_path, "w") as f: 
        for style, idx in style_to_idx.items():
            f.write(f"{style}: {idx}\n")
            
    #create dataloaders 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    #move model to device 
    model = model.to(device)
    print(f"Model: {model}")
    
    #define loss functions 
    transcription_loss_fn = nn.CrossEntropyLoss(ignore_index=-100) #note to self: looking to the special tokens in the whisper processor 
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
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar: 
            #move batch to device 
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            timestamp_indices = batch["timestamp_indices"]
            emotion_labels = [el.to(device) for el in batch["emotion_labels"]]
            
            #zero gradients 
            optimizer.zero_grad()
            
            #forward pass 
            outputs = model(
                input_features=input_features, 
                decoder_input_ids=labels[:, :-1] if labels.size(1) > 1 else None, #teacher forcing 
                timestamp_indices=timestamp_indices
            )
            
            logits = outputs["logits"]
            emotion_logits = outputs["emotion_logits"]
            
            #calculate transcription loss 
            shifted_logits = logits[:, :-1, :].contiguous() if logits.size(1) > 1 else logits #this isnt teacher forcing, apparently whisper adds an additional token that we have to remove -> double check this  
            shifted_labels = labels[:, 1:].contiguous() if labels.size(1) > 1 else labels #teacher forcing 
            
            #calculate transcription loss 
            transcription_loss = transcription_loss_fn(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
            
            #calculate emotion loss (per segment)
            emotion_loss, emotion_correct, emotion_total = calculate_emotion_loss(emotion_logits, emotion_labels, emotion_loss_fn)
            
            #accumulate metrics 
            train_emotion_correct += emotion_correct
            train_emotion_total += emotion_total 
            
            #calculate total loss 
            loss = transcription_loss + args.emotion_weight * emotion_loss 
                      
            #backprop + update weights 
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            #update progress bar 
            train_loss += loss.item()
            emotion_acc = train_emotion_correct / train_emotion_total if train_emotion_total > 0 else 0
            progress_bar.set_postfix(
                "loss": loss.item(), 
                "tr_loss": transcription_loss.item(), 
                "emo_loss": emotion_loss.item() if total > 0 else 0, 
                "emo_acc": f"{emotion_acc:.2f}"
            )
            
            #calculate average training loss and accuracy 
            avg_train_loss = train_loss / len(train_loader)
            train_emotion_acc = train_emotion_correct / train_emotion_total if train_emotion_total > 0 else 0 
            print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Emotion Segment Accuracy: {train_emotion_acc:.2f}")
            
            
            #Validation 
            model.eval()
            val_loss = 0.0 
            val_emotion_correct = 0 
            val_emotion_total = 0 
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc="Validation")
                
                for batch in progress_bar: 
                    #move batch to device 
                    input_features = batch["input_features"].to(device)
                    labels = batch["labels"].to(device)
                    timestamp_indices = batch["timestamp_indices"]
                    emotion_labels = [el.to(device) for el in batch["emotion_labels"]]
                    
                    #forward pass 
                    outputs = model(
                        input_features=input_features, 
                        decoder_input_ids=labels[:, :-1] if labels.size(1) > 1 else None, 
                        timestamp_indices=timestamp_indices
                    )
                    
                    logits = outputs["logits"]
                    emotion_logits = outputs["emotion_logits"]
                    
                    #calculate the transcription loss 
                    shifted_logits = logits[:, :-1, :].contiguous() if logits.size(1) > 1 else logits 
                    shifted_labels = labels[:, 1:].contiguous() if labels.size(1) > 1 else labels 
                    
                    #calculate transcription loss 
                    transcription_loss = transcription_loss_fn(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
                    
                    #calculate emotion loss and accuracy 
                    emotion_loss, emotion_correct, emotion_total = calculate_emotion_loss(emotion_logits, emotion_labels, emotion_loss_fn)
                    
                    # accumulate metrics 
                    val_emotion_correct += emotion_correct
                    val_emotion_total += emotion_total 
                    
                    #calculate total loss 
                    loss = transcription_loss + args.emotion_weight * emotion_loss 
                    
                    #accumulate loss 
                    val_loss += loss.item()
                    
                    
            #calculate average validation loss and segment accuracy 
            avg_val_loss = val_loss / len(val_loader)
            val_emotion_acc = val_emotion_correct / val_emotion_total if val_emotion_total > 0 else 0 
            print(f"Epoch {epoch+1}/{args.num_epochs}, Val Loss: {avg_val_loss:.4f}, Emotion Segment Accuracy: {val_emotion_acc:.2f}")
            
            #save model if better than best 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss 
                model_path = os.path.join(args.output_dir, f"best_model_epoch{epoch+1}.pth")
                os.makedirs(model_path, exist_ok=True)
                model.save_pretrained(model_path)
                print(f"Saved best model to {model_path}")
        
    #save final model 
    final_model_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    train()