import os 
import argparse 
import torch 
import torch.nn as nn 
import torch.optim as optim 

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
            features, labes = features.to(device), labels.to(device)
            
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
    
    #create output directory 
    os.makedirs(args.output_dir, exist_ok=True)
    
    #load dataset 
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, 
        sr=args.sample_rate, 
        duration=args.duration, 
        n_mels=args.n_mels
    )
    
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
    best_val_accuracy = 0.0 
    
    for epoch in range(args.epochs):
        #train 
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        #validate 
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        #update learning rate 
        scheduler.step(val_loss)
    
        
        #print metrics
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        #save best model 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model (accuracy: {best_val_acc:.2f}%)")
            
        return model, best_val_acc
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Urban Sound Classification Training')
    
    #dataset parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_mels', type=int, default=64)
    
    #model parameters 
    parser.add_argument('--conv_dim', type=int, default=128)
    parser.add_argument('--encoder_dim', type=int, default=256)
    parser.add_argument('--num_encoder_blocks', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    
    #training parameters 
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    
    # Other parameters
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--no_cuda", action="store_true")
    
    args = parser.parse_args()
    
    # Train the model
    model, best_acc = train_classifier(args)
    
        
        
