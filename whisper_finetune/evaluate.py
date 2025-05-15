import os 
import argparse 
import torch 
from tqdm import tqdm 
from torch.utils.data import DataLoader 
from torch.utils.data import DataLoader 
from sklearn.metrics import accuracy_score, f1_score, classification_report 


from model import load_emotion_whisper_model 
from dataset import create_dataset, SIMPLE_STYLES 

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Emotion Whisper model")
    parser.add_argument("--model_path", type=str, default="./emotion_whisper_model/best_model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--simple_styles", action="store_true", help="Use simplified emotion styles instead of full set")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    return parser.parse_args()


def get_segments_with_timestamps(processor, model, input_features, device):
    """Generate trasncription with timestamps and extract segments"""
    #run inference with forced timestamp prediction 
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        task="transcribe", 
        language="en", 
        return_timestamps=True
    ) #shape: [{token_id_key: token_id_value, token_id_key: token_id_value, ...}]
    
    with torch.no_grad():
        outputs = model.whisper.generate(
            input_features.to(device), 
            forced_decoder_ids=forced_decoder_ids, 
            return_dict_ingenerate=True,
            max_length=256)
        
    #process generated sequence 
    timestamp_tokens = []
    segments = []
    
    for batch_idx, seq in enumerate(outputs.sequences): 
        #outputs.sequence: [B, generated_sequence_length]
        #seq: [generated_sequence_length]
        
        tokens = seq.cpu().numpy()
        seq_timestamps = []
        seq_segments = []
        current_segment = ""
        segment_start = 0.0 
        
        #process tokens to extract timestamps 
        for i, token_id in enumerate(tokens): 
            #check for timestamp tokens 
            if token_id >= processor.tokenizer.timestamp_begin: 
                timestamp_value = float(token_id = processor.tokenizer.timestamp_begin) / 50.0 
                if timestamp_value > segment_start: 
                    seq_segments.append({
                        "text": processor.tokenizer.decode(seq[:i], skip_special_tokens=True).strip(),
                        "start": segment_start, 
                        "end": timestamp_value
                    })
                    seq_timestamps.append(i)
                segment_start = timestamp_value 
        #store results 
        segments.append(seq_segments)
        timestamp_tokens.append(seq_timestamps)  
    
    return segments, timestamp_tokens 

def main(): 
    args = parse_args()
    
    #create output directory 
    os.makedirs(args.output_dir, exist_ok=True)
    
    #set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #load model and processor 
    model,processor = load_emotion_whisper_model()
    model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))
    model = model.to(device)
    model.eval()
    
    #load test dataset 
    print("loading test dataset...")
    selected_styles = SIMPLE_STYLES if args.simple_styles else None 
    _,_, test_dataset, style_to_idx = create_dataset(
        processor=processor, 
        selected_styles=selected_styles
    ) 
    
    #create dataloader 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=test_dataset.collate_fn
    )
    
    #prepare for evaluation 
    all_true_emotions = []
    all_pred_emotions = []
    total_segments = 0
    
    #evaluate segment by segment 
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_features = batch["input_features"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            
            #generate segments using whisper timestamps 
            segments, timestamp_tokens = get_segments_with_timestamps(
                processor, model, input_features, device 
            )
            
            #classify emotions for each segment 
            for b in range(len(segments)):
                #skip empty segments 
                if not segments[b]:
                    continue 
                
                #run model with timestamp indices for segment-based prediction 
                outputs = model(
                    input_features=input_features[b:b+1],
                    timestamp_indices=[timestamp_tokens[b]]
                )
                
                #get emotion preidctions for each segment 
                seq_emotion_logits = outputs["emotion_logits"][0] #first batch item 
                if seq_emotion_logits.dim() == 1: #only one segment 
                    seq_emotion_logits = seq_emotion_logits.unsqueeze(0)
                
                #get predicted emotions for each segment 
                pred_emotions = torch.argmax(seq_emotion_logits, dim=1).cpu().numpy().tolist()
                
                #use sequence-level emotion as ground truth for all segements 
                true_emotion = emotion_labels[b].item()   
                true_emotions = [true_emotion] * len(pred_emotions)  
                
                #store results 
                all_true_emotions.extend(true_emotions)
                all_pred_emotions.extend(pred_emotions)         
                total_segments += len(pred_emotions)
    
    #calculate metrics 
    accuracy = accuracy_score(all_true_emotions, all_pred_emotions)
    f1 = f1_score(all_true_emotions, all_pred_emotions, average="weighted")
    
    #get class names for report 
    idx_to_style = {idx: style for style, idx in style_to_idx.items()}
    
    print(f"Total segments evaluated: {total_segments}")
    print(f"Segment-level Emotion Classification Accuracy: {accuracy:.4f}")
    print(f"Segment-level Emotion Classification F1 Score: {f1:.4f}")
    
    #generate classification report 
    report = classification_report(
        all_true_emotions, 
        all_pred_emotions, 
        labels=sorted(list(set(all_true_emotions + all_pred_emotions))),
        target_names=[idx_to_style.get(i, f"Unknown_{i}") for i in sorted(list(set(all_true_emotions + all_pred_emotions)))],
        digits = 3
    )
    
    print("Classification Report:")
    print(report)
    
    # Save results
    with open(os.path.join(args.output_dir, "segment_evaluation.txt"), "w") as f:
        f.write(f"Total segments evaluated: {total_segments}\n")
        f.write(f"Segment-level Emotion Classification Accuracy: {accuracy:.4f}\n")
        f.write(f"Segment-level Emotion Classification F1 Score: {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    
    print(f"Results saved to {os.path.join(args.output_dir, 'segment_evaluation.txt')}")


    if __name__ == "__main__":
        main()
        
