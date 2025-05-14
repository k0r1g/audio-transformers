import os 
import argparse 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import wandb
from huggingface_hub import HfApi

from dataset import get_dataloaders
from model import UrbanSoundModel

def train_epoch(model, dataloader, criterion, optimizer, device):
    """train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0 
    total = 0 
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        #forward pass 
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        #backward pass 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #update running loss and accuracy 
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0) #labels has shape [batch_size]
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    """evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0 
    total = 0 
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            #update running loss and accuracy 
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return running_loss / len(dataloader), 100. * correct / total

def train_classifier(args):
    #setup device 
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(project="mlx-audio-models", name=args.run_name)
    wandb.config.update(args)
    
    #create output directory 
    os.makedirs(args.output_dir, exist_ok=True)
    
    #load dataset 
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, 
        sr=args.sample_rate, 
        duration=args.duration, 
        n_mels=args.n_mels,
        limit_samples=args.limit_samples
    )
    
    print(f"Training with {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")

    
    #create model 
    model = UrbanSoundModel(
        n_mels=args.n_mels, 
        num_classes=10, 
        conv_dim=args.conv_dim, 
        encoder_dim=args.encoder_dim, 
        num_encoder_blocks=args.num_encoder_blocks, 
        num_heads=args.num_heads, 
    )
    
    model = model.to(device)
    
    #loss function and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    #training loop 
    best_val_acc = 0.0 
    best_model_path = None
    
    for epoch in range(args.epochs):
        #train 
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        #validate 
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        #update learning rate 
        scheduler.step(val_loss)
    
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        #print metrics
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        #save best model 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (accuracy: {best_val_acc:.2f}%)")
    
    # Upload best model to HuggingFace Hub
    if best_model_path and args.upload_to_hub:
        print(f"Uploading best model to HuggingFace Hub: {args.hf_repo_id}")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=best_model_path,
            path_in_repo="best_model.pt",
            repo_id=args.hf_repo_id,
            token=os.environ.get("HF_ACCESS")
        )
        print(f"Successfully uploaded model to {args.hf_repo_id}")
    
    # Finish wandb run
    wandb.finish()
            
    return model, best_val_acc
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Urban Sound Classification Training')
    
    #dataset parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--duration', type=float, default=4.0)
    parser.add_argument('--limit_samples', type=int, default=None, help='Limit the number of samples in train and test sets')
    
    #model parameters 
    parser.add_argument('--conv_dim', type=int, default=128)
    parser.add_argument('--encoder_dim', type=int, default=256)
    parser.add_argument('--num_encoder_blocks', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    
    #training parameters 
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    
    # Integration parameters
    parser.add_argument('--upload_to_hub', action='store_true', help='Upload model to HuggingFace Hub')
    parser.add_argument('--hf_repo_id', type=str, default='Kogero/urbansound8kclassifier', help='HuggingFace repository ID')
    parser.add_argument('--run_name', type=str, default='urbansound-training', help='Name for the wandb run')
    
    # Other parameters
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--no_cuda", action="store_true")
    
    args = parser.parse_args()
    
    # Train the model
    model, best_acc = train_classifier(args)
    
        
        
