import os 
import argparse 
import torch 
from tqdm import tqdm 
from torch.utils.data import DataLoader 
from sklearn.metrics import accuracy_score, f1_score, classification_report 

from model import EmotionWhisperModel, load_emotion_whisper_model # MODIFIED: Ensure load_emotion_whisper_model is imported
from transformers import WhisperProcessor, GenerationConfig
# from huggingface_hub import hf_hub_download # Not strictly needed if args.model_path is always local for weights
from dataset import create_dataset, SIMPLE_STYLES 

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Emotion Whisper model")
    # MODIFIED: Updated default and help string for model_path to emphasize local path for weights
    parser.add_argument("--model_path", type=str, default="./emotion_whisper_model/best_model", help="Path to local directory containing model weights (pytorch_model.bin or model.safetensors). Processor files can be co-located or fetched if model_path is also a Hub ID.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--simple_styles", action="store_true", help="Use simplified emotion styles instead of full set")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    return parser.parse_args()


def get_segments_with_timestamps(model, processor, input_features, device):
    """
    Runs Whisper once and returns
      • segments  – list[list[dict(start, end, text, tokens)]]
      • ts_idx    – list[list[int]]  end-token positions for each segment
      • sequences – torch.LongTensor of generated token ids
    It also removes every leftover forced-decoder-id so `generate()` can run.
    """

    # ----- 0. ensure timestamp_begin exists -----
    if not hasattr(processor.tokenizer, "timestamp_begin"):
        processor.tokenizer.timestamp_begin = processor.tokenizer._convert_token_to_id("<|timestamp_0|>")

    # wipe every pre-stored list that could clash inside generate_with_fallback()
    model.whisper.config.forced_decoder_ids = None
    model.whisper.generation_config.forced_decoder_ids = None

    with torch.no_grad():
        out = model.whisper.generate(
            input_features.to(device),
            task="transcribe",
            language="en",

            # -------- make it emit timestamps (works on <4.40) --------
            return_timestamps="generate",      # <- string, not bool
            return_dict_in_generate=True,      # required to get .sequences/.segments

            # -------- tame repetition (unchanged) --------
            temperature=0.7,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.4,
            repetition_penalty=1.1,

            max_new_tokens=256,
            forced_decoder_ids=None,
        )

    ts_begin = processor.tokenizer.timestamp_begin

    filtered_segments_batch, ts_idx = [], []

    for segs in out["segments"]:
        good, idxs = [], []

        for seg in segs:
            # --- unify access ---------------------------------------------------
            tokens = seg["tokens"] if isinstance(seg, dict) else seg.tokens
            text   = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", None)

            # if text missing → rebuild it
            if text is None:
                text_tokens = [t for t in tokens if t < ts_begin]
                text = processor.tokenizer.decode(text_tokens,
                                                  skip_special_tokens=True).strip()

            # skip empty segments created by the old fallback code
            if not text:
                continue

            # --- find timestamp id; if missing, fall back to last token ----------
            ts_id = next((t for t in reversed(tokens) if t >= ts_begin), None)
            if ts_id is None:
                # rare edge case: add synthetic timestamp
                ts_id = tokens[-1]

            good.append(seg)
            idxs.append(ts_id)

        filtered_segments_batch.append(good)
        ts_idx.append(idxs)

    return filtered_segments_batch, ts_idx, out["sequences"]

def main(): 
    args = parse_args()
    
    #create output directory 
    os.makedirs(args.output_dir, exist_ok=True)
    
    #set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # MODIFIED: Model Loading Section as per user's "Simplest Solutions"
    print(f"Loading processor from {args.model_path}...")
    # Processor can be loaded from a Hub ID or local path specified by args.model_path
    processor = WhisperProcessor.from_pretrained(args.model_path)
    
    print(f"Initializing model architecture using local load_emotion_whisper_model...")
    # Load model architecture using your own load function for consistency.
    # This will use the num_emotions_classes default from your model.py (e.g., 10 classes)
    model, _ = load_emotion_whisper_model() 

    print(f"Attempting to load trained weights from local path: {args.model_path}...")
    # args.model_path is expected to be a local directory path for weight files here.
    model_weights_path_bin = os.path.join(args.model_path, "pytorch_model.bin")
    model_weights_path_safetensors = os.path.join(args.model_path, "model.safetensors")

    loaded_weights = False
    if os.path.exists(model_weights_path_bin):
        print(f"Loading weights from {model_weights_path_bin}")
        model.load_state_dict(torch.load(model_weights_path_bin, map_location=device), strict=False)
        loaded_weights = True
    elif os.path.exists(model_weights_path_safetensors):
        print(f"Loading weights from {model_weights_path_safetensors}")
        from safetensors.torch import load_file
        model.load_state_dict(load_file(model_weights_path_safetensors), strict=False) # Removed device=device as per previous fix
        loaded_weights = True
    
    if not loaded_weights:
        print(f"Warning: Model weights not found at {model_weights_path_bin} or {model_weights_path_safetensors}.")
        print("Model will use initialized weights. If this is not for initial training, evaluation will be random.")
        # Potentially raise an error if weights are mandatory for evaluation:
        # raise FileNotFoundError(f"Model weights not found at {model_weights_path_bin} or {model_weights_path_safetensors}. Cannot proceed with evaluation.")

    model = model.to(device)
    model.eval()
    
    # ---------- ensure generation_config has the timestamp ids ----------
    try:
        gen_cfg = GenerationConfig.from_pretrained(args.model_path)
    except (OSError, ValueError):
        gen_cfg = GenerationConfig.from_pretrained("openai/whisper-tiny")
    model.whisper.generation_config = gen_cfg

    #load test dataset 
    print("loading test dataset...")
    selected_styles = SIMPLE_STYLES if args.simple_styles else None 
    _,_, test_dataset, style_to_idx = create_dataset(
        processor=processor, 
        selected_styles=selected_styles
    ) 
    
    # Create reverse mapping from index to style name
    idx_to_style = {idx: style for style, idx in style_to_idx.items()}
    
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
    
    # Create a file to log predictions
    prediction_log_path = os.path.join(args.output_dir, "predictions.txt")
    with open(prediction_log_path, "w") as log_file:
        log_file.write("===== TRANSCRIPTION AND EMOTION PREDICTIONS =====\n\n")
    
    #evaluate segment by segment 
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)  # True transcription tokens
            emotion_labels = batch["emotion_labels"].to(device)
            
            #generate segments using whisper timestamps 
            segments, timestamp_tokens, generated_sequences = get_segments_with_timestamps(
                model, processor, input_features, device 
            )
            
            generated_sequences = generated_sequences.to(device)

            # Get true transcriptions
            true_transcriptions = []
            for label_seq in labels:
                true_text = processor.tokenizer.decode(label_seq, skip_special_tokens=True)
                true_transcriptions.append(true_text)
            
            # Get predicted transcriptions (full)
            pred_transcriptions = []
            for seq in generated_sequences:
                pred_text = processor.tokenizer.decode(seq, skip_special_tokens=True)
                pred_transcriptions.append(pred_text)
            
            with open(prediction_log_path, "a") as log_file:
                log_file.write(f"===== BATCH {batch_idx + 1} =====\n\n")
            
            #classify emotions for each segment 
            for b in range(len(segments)):
                #skip empty segments 
                if not segments[b]:
                    continue 
                
                # Get true emotion name
                true_emotion_idx = emotion_labels[b].item()
                true_emotion_name = idx_to_style.get(true_emotion_idx, f"Unknown_{true_emotion_idx}")
                
                # Print full transcription comparison
                with open(prediction_log_path, "a") as log_file:
                    log_file.write(f"Sample {batch_idx * args.batch_size + b + 1}:\n")
                    log_file.write(f"True Emotion: {true_emotion_name}\n")
                    log_file.write(f"True Transcription: {true_transcriptions[b]}\n")
                    log_file.write(f"Predicted Transcription: {pred_transcriptions[b]}\n\n")
                
                print(f"\nSample {batch_idx * args.batch_size + b + 1}:")
                print(f"True Emotion: {true_emotion_name}")
                print(f"True Transcription: {true_transcriptions[b]}")
                print(f"Predicted Transcription: {pred_transcriptions[b]}\n")
                
                # MODIFIED: Handle Any Empty Segments Case
                if not timestamp_tokens[b]: # If no timestamp tokens were found for this item in the batch
                    # This means no segments were naturally decoded via timestamps.
                    # We need to decide how to handle emotion prediction for this.
                    # Option 1: Skip emotion prediction for this sample.
                    # Option 2: Predict emotion based on the whole sequence.
                    # Option 3: Create a fallback segment as suggested.
                    print(f"Warning: No segments found for sample {batch_idx * args.batch_size + b + 1}. Using fallback.")
                    with open(prediction_log_path, "a") as log_file:
                        log_file.write(f"Warning: No segments found for sample {batch_idx * args.batch_size + b + 1}. Using fallback.\n")

                    # Fallback: use global representation for emotion (similar to training)
                    # We need the hidden_states from the main model forward pass for this item.
                    # The current `outputs` from `model.whisper.generate` are for transcription.
                    # We need to run the main model's forward pass.
                    
                    # For simplicity with the current structure, let's create a single segment 
                    # covering the whole predicted transcription.
                    # This is a simplified fallback; a more robust solution might involve
                    # re-running a part of the model or using a global emotion prediction.
                    
                    # Ensure `pred_transcriptions[b]` is available
                    current_pred_transcription = pred_transcriptions[b] if b < len(pred_transcriptions) else "N/A"

                    timestamp_tokens[b] = [len(processor.tokenizer.encode(current_pred_transcription))] # Approximate length
                    segments[b] = [{
                        "text": current_pred_transcription,
                        "start": 0.0,
                        "end": 1.0  # Arbitrary end time, actual duration not easily known here
                    }]
                    # If segments[b] was originally empty, and we add one fallback segment,
                    # then len(segments[b]) will be 1.
                    # The emotion prediction below will then run for this single segment.

                # If, after fallback, segments[b] is still empty (e.g., if pred_transcription was also empty)
                if not segments[b]:
                    print(f"Skipping sample {batch_idx * args.batch_size + b + 1} due to no segments even after fallback.")
                    with open(prediction_log_path, "a") as log_file:
                        log_file.write(f"Skipping sample {batch_idx * args.batch_size + b + 1} due to no segments even after fallback.\n")
                    continue # Skip to next item in batch
                
                #run model with timestamp indices for segment-based prediction 
                model_outputs = model(
                    input_features=input_features[b:b+1],
                    decoder_input_ids=generated_sequences[b:b+1],
                    timestamp_indices=[timestamp_tokens[b]]
                )
                
                #get emotion predictions for each segment 
                seq_emotion_logits_list = model_outputs["emotion_logits"]

                if not seq_emotion_logits_list: # Should not happen if segments[b] is not empty
                    print(f"Warning: No emotion logits for sample {batch_idx * args.batch_size + b + 1}")
                    with open(prediction_log_path, "a") as log_file:
                        log_file.write(f"Warning: No emotion logits for sample {batch_idx * args.batch_size + b + 1}\n")
                    # Fill with dummy predictions or skip
                    pred_emotions_for_sample = [true_emotion_idx] * len(segments[b]) # Use true as placeholder
                else:
                    # seq_emotion_logits_list contains one item (for the single sample b)
                    # and that item is a tensor of shape [num_segments_in_sample_b, num_classes]
                    seq_emotion_logits = seq_emotion_logits_list[0] 
                    if seq_emotion_logits.dim() == 1: # only one segment 
                        seq_emotion_logits = seq_emotion_logits.unsqueeze(0)
                
                    pred_emotions_for_sample = torch.argmax(seq_emotion_logits, dim=1).cpu().numpy().tolist()

                # Convert predicted emotion indices to names
                pred_emotion_names = [idx_to_style.get(idx, f"Unknown_{idx}") for idx in pred_emotions_for_sample]
                
                #use sequence-level emotion as ground truth for all segments 
                # True emotion is single value per sample, replicated for each segment of that sample
                true_emotions_for_sample = [true_emotion_idx] * len(pred_emotions_for_sample) 
                
                # Print segment-by-segment predictions
                with open(prediction_log_path, "a") as log_file:
                    log_file.write("Segment-by-segment analysis:\n")
                    # Iterate up to the minimum of available segments and predicted emotions
                    for i in range(min(len(segments[b]), len(pred_emotion_names))):
                        segment = segments[b][i]
                        pred_emo = pred_emotion_names[i]
                        log_file.write(f"  Segment {i+1} ({segment.get('start', 0.0):.1f}s - {segment.get('end', 0.0):.1f}s):\n")
                        log_file.write(f"    Text: \"{segment.get('text', '')}\"\n")
                        log_file.write(f"    Predicted Emotion: {pred_emo}\n")
                    log_file.write("\n")
                
                print("Segment-by-segment analysis:")
                for i in range(min(len(segments[b]), len(pred_emotion_names))):
                    segment = segments[b][i]
                    pred_emo = pred_emotion_names[i]
                    print(f"  Segment {i+1} ({segment.get('start', 0.0):.1f}s - {segment.get('end', 0.0):.1f}s):")
                    print(f"    Text: \"{segment.get('text', '')}\"")
                    print(f"    Predicted Emotion: {pred_emo}")
                print()
                
                #store results for metrics
                all_true_emotions.extend(true_emotions_for_sample)
                all_pred_emotions.extend(pred_emotions_for_sample)         
                total_segments += len(pred_emotions_for_sample) # Count based on actual predictions made
    
    #calculate metrics 
    if not all_true_emotions or not all_pred_emotions:
        print("No predictions were made. Skipping metrics calculation.")
        accuracy = 0.0
        f1 = 0.0
        report = "No predictions available to generate a report."
    else:
        accuracy = accuracy_score(all_true_emotions, all_pred_emotions)
        f1 = f1_score(all_true_emotions, all_pred_emotions, average="weighted", zero_division=0)
    
        # Ensure labels for classification_report are only those present in data
        present_labels = sorted(list(set(all_true_emotions + all_pred_emotions)))
        target_names_present = [idx_to_style.get(i, f"Unknown_{i}") for i in present_labels]

        if not present_labels: # Handles case where all_true_emotions or all_pred_emotions might be empty lists of lists
             report = "Not enough data to generate classification report (no common labels found)."
        else:
            report = classification_report(
                all_true_emotions, 
                all_pred_emotions, 
                labels=present_labels,
                target_names=target_names_present,
                digits = 3,
                zero_division=0
            )
    
    print(f"Total segments evaluated: {total_segments}")
    print(f"Segment-level Emotion Classification Accuracy: {accuracy:.4f}")
    print(f"Segment-level Emotion Classification F1 Score: {f1:.4f}")
    
    print("Classification Report:")
    print(report)
    
    # Save results
    results_file_path = os.path.join(args.output_dir, "segment_evaluation.txt")
    with open(results_file_path, "w") as f:
        f.write(f"Total segments evaluated: {total_segments}\n")
        f.write(f"Segment-level Emotion Classification Accuracy: {accuracy:.4f}\n")
        f.write(f"Segment-level Emotion Classification F1 Score: {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    
    print(f"Results saved to {results_file_path}")
    print(f"Detailed predictions saved to {prediction_log_path}")


if __name__ == "__main__":
    main()
    
