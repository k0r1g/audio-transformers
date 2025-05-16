import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import WhisperProcessor, GenerationConfig
from model import EmotionWhisperModel # Assuming model.py is in the same directory or accessible
import os

DEFAULT_MODEL_PATH = "./emotion_whisper_model/best_model_epoch7" # Default, can be overridden

def load_model_and_processor(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads the EmotionWhisperModel and WhisperProcessor from the specified path.
    Also handles weight tying and generation config setup.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference: Using device: {device}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path {model_path} not found. Please ensure the model is trained and saved correctly or provide a valid path.")

    try:
        processor = WhisperProcessor.from_pretrained(model_path)
        model = EmotionWhisperModel.from_pretrained(model_path)
        print(f"Successfully loaded model and processor from: {model_path}")
    except Exception as e:
        print(f"Error loading model/processor from {model_path}. Trying openai/whisper-tiny as fallback for processor if path was a Hub ID for a non-Whisper model.")
        # This fallback for processor might be too generic if model_path itself was wrong.
        # Consider if model_path is a Hub ID, from_pretrained for EmotionWhisperModel would handle it.
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny") # Fallback if processor loading fails from model_path
        model = EmotionWhisperModel.from_pretrained(model_path) # This would handle Hub ID if model_path points to it
        print(f"Successfully loaded model from {model_path} (processor may be fallback). Error was: {e}")


    # Tie weights
    with torch.no_grad():
        if hasattr(model, 'whisper') and \
           hasattr(model.whisper, 'proj_out') and \
           hasattr(model.whisper, 'model') and \
           hasattr(model.whisper.model, 'decoder') and \
           hasattr(model.whisper.model.decoder, 'embed_tokens') and \
           model.whisper.proj_out is not None and \
           model.whisper.model.decoder.embed_tokens is not None:
            model.whisper.proj_out.weight = model.whisper.model.decoder.embed_tokens.weight
            print("✓ proj_out.weight tied to decoder embeddings")
        else:
            print("⚠️ Warning: Could not tie proj_out.weight. Required attributes not found or are None.")

    # Setup GenerationConfig
    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path)
        print(f"✓ generation_config.json found with the model at {model_path}")
    except (OSError, ValueError):
        gen_cfg = GenerationConfig.from_pretrained("openai/whisper-tiny")
        print(f"➜ No generation_config.json found with the model at {model_path} – borrowed one from openai/whisper-tiny")

    gen_cfg.forced_decoder_ids = None # Clear any stored forced_decoder_ids

    if hasattr(model, "whisper") and model.whisper is not None:
        model.whisper.config.forced_decoder_ids = None
        model.whisper.generation_config = gen_cfg # Re-attach the modified config
    else:
        print("Warning: model.whisper attribute not found or is None during gen_cfg setup.")
        
    model.to(device).eval()
    return model, processor, device

def load_emotion_labels(model_path: str = DEFAULT_MODEL_PATH) -> list:
    """
    Loads emotion labels from style_to_id.txt in the model directory.
    If the file is not found or is invalid, falls back to a predefined list.
    Returns an ordered list of emotion names.
    """
    style_map_path = Path(model_path) / "style_to_id.txt"
    idx_to_style = {}
    ordered_labels = []

    # Define the fallback labels based on the provided mapping
    # Order is crucial and derived from the indices: 0:confused, 1:default, ...
    fallback_emotion_labels = [
        "confused", "default", "emphasis", "enunciated", "essentials", 
        "happy", "laughing", "sad", "singing", "whisper"
    ]

    if style_map_path.exists():
        try:
            with open(style_map_path, "r") as f:
                for line in f:
                    if ":" in line:
                        style, idx_str = line.strip().split(":", 1)
                        idx = int(idx_str.strip())
                        idx_to_style[idx] = style.strip()
            
            if idx_to_style:
                # Ensure the list is ordered by index
                max_idx = -1
                try:
                    # Attempt to find the highest index to determine list size
                    if idx_to_style:
                         max_idx = max(idx_to_style.keys())
                    ordered_labels = [""] * (max_idx + 1)
                    for idx, style in idx_to_style.items():
                        if idx < len(ordered_labels):
                            ordered_labels[idx] = style
                        else:
                            # This case should ideally not happen if idx_to_style is contiguous from 0
                            print(f"Warning: Index {idx} out of bounds for label list of size {len(ordered_labels)}. Label '{style}' might be misplaced or ignored.")
                    # Filter out any empty strings if there were gaps, though ideally there shouldn't be
                    ordered_labels = [label for label in ordered_labels if label] 

                    if len(ordered_labels) == len(fallback_emotion_labels):
                        print(f"Successfully loaded and ordered emotion labels from {style_map_path}: {ordered_labels}")
                        return ordered_labels
                    else:
                        print(f"Warning: Labels loaded from {style_map_path} ({len(ordered_labels)} labels: {ordered_labels}) do not match expected count ({len(fallback_emotion_labels)}). Using fallback.")
                except Exception as e:
                    print(f"Error ordering labels from {style_map_path}: {e}. Using fallback.")
            else:
                print(f"No valid labels parsed from {style_map_path}. Using fallback.")
        except Exception as e:
            print(f"Error reading or parsing {style_map_path}: {e}. Using fallback.")
    else:
        print(f"Warning: style_to_id.txt not found at {style_map_path}. Using fallback labels.")

    print(f"Using fallback emotion labels: {fallback_emotion_labels}")
    return fallback_emotion_labels

def perform_inference(audio_data: np.ndarray, 
                      sampling_rate: int,
                      model: EmotionWhisperModel, 
                      processor: WhisperProcessor, 
                      device: torch.device,
                      segment_duration: int = 5):
    """
    Performs transcription and segmented emotion analysis on the given audio data.

    Args:
        audio_data (np.ndarray): The audio waveform.
        sampling_rate (int): The sampling rate of the audio.
        model: The loaded EmotionWhisperModel.
        processor: The loaded WhisperProcessor.
        device: The torch device to use.
        segment_duration (int): Duration of segments for emotion analysis in seconds.

    Returns:
        tuple: (full_transcription_text, list_of_segment_emotion_probs_df)
               Each element in list_of_segment_emotion_probs_df is a pandas DataFrame
               with 'Emotion' and 'Probability' columns for a segment.
    """

    # 1. Full Transcription
    print("Performing full transcription...")
    input_features_full = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
    
    full_transcription_text = ""
    if hasattr(model, 'whisper') and model.whisper is not None:
        with torch.no_grad():
            generated_ids_full = model.whisper.generate(
                input_features_full,
                max_new_tokens=128, # Using a reasonable default, consider making configurable
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                length_penalty=-0.5, # From evaluate_simple
                forced_decoder_ids=None # Crucial from evaluate_simple fixes
            )
        full_transcription_text = processor.decode(generated_ids_full[0], skip_special_tokens=True)
        print(f"Full Transcription: {full_transcription_text}")
    else:
        print("Error: model.whisper not found. Cannot generate full transcription.")


    # 2. Segmented Emotion Analysis
    print(f"Performing segmented emotion analysis (segment_duration={segment_duration}s)...")
    num_samples_total = len(audio_data)
    samples_per_segment = sampling_rate * segment_duration
    num_segments = int(np.ceil(num_samples_total / samples_per_segment))
    
    segment_emotion_probs_list = []

    if num_segments == 0 and num_samples_total > 0:
        num_segments = 1 # Handle very short audio

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = min((i + 1) * samples_per_segment, num_samples_total)
        segment_audio_data = audio_data[start_sample:end_sample]

        if len(segment_audio_data) == 0:
            print(f"Segment {i+1} is empty, skipping.")
            # Add placeholder for empty segment if necessary, or just skip
            # For now, let's ensure the list length matches num_segments if needed by UI
            # Or better, only add valid results. For simplicity now, let's say UI handles varying list length.
            continue

        print(f"Processing segment {i+1}/{num_segments}...")
        segment_input_features = processor(segment_audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

        segment_emotion_logits = None
        if hasattr(model, 'whisper') and model.whisper is not None:
            with torch.no_grad():
                # Generate transcription tokens for THIS segment to feed into the emotion head
                # Use shorter max_new_tokens for segments
                segment_generated_ids = model.whisper.generate(
                    segment_input_features,
                    max_new_tokens=64, # Shorter for segments
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1, # Slightly less penalty for short segments
                    forced_decoder_ids=None 
                )
                
                # Get emotion logits for the segment
                # The model's forward pass will use global pooling over the segment's hidden states
                outputs = model(input_features=segment_input_features, decoder_input_ids=segment_generated_ids)
                segment_emotion_logits = outputs["emotion_logits"]
        else:
            print(f"Error: model.whisper not found. Cannot generate for segment {i+1}.")
            segment_emotion_probs_list.append(None) # Or some indicator of failure
            continue
            
        if segment_emotion_logits is not None:
            # Apply softmax to get probabilities
            segment_emotion_probs = torch.softmax(segment_emotion_logits, dim=-1).squeeze().cpu().numpy()
            segment_emotion_probs_list.append(segment_emotion_probs)
            print(f"Segment {i+1} emotion probabilities (raw shape): {segment_emotion_probs.shape}")
        else:
            print(f"Segment {i+1} emotion logits are None.")
            segment_emotion_probs_list.append(None)


    return full_transcription_text, segment_emotion_probs_list

if __name__ == '__main__':
    # Example Usage (assuming you have an audio file 'test_audio.wav')
    print("Running inference.py example...")
    
    # Create a dummy audio file for testing if it doesn't exist
    dummy_audio_path = "dummy_test_audio.wav"
    if not os.path.exists(dummy_audio_path):
        print(f"Creating dummy audio file: {dummy_audio_path}")
        sr_dummy = 16000
        duration_dummy = 12 # seconds, for a few segments
        # Create a simple sine wave
        frequency = 440  # A4 note
        t = np.linspace(0, duration_dummy, int(sr_dummy * duration_dummy), False)
        audio_dummy = 0.5 * np.sin(2 * np.pi * frequency * t)
        # Add some noise to make it less uniform for Whisper
        noise = 0.01 * np.random.randn(len(audio_dummy))
        audio_dummy += noise
        
        import soundfile as sf
        sf.write(dummy_audio_path, audio_dummy, sr_dummy)
        print("Dummy audio file created.")

    model_path_to_test = DEFAULT_MODEL_PATH 
    # Check if the default model path exists, otherwise skip example run or error
    if not Path(model_path_to_test).exists():
        print(f"Default model path {model_path_to_test} does not exist. Skipping example run.")
        print("Please train a model or provide a valid --model_path to evaluate_simple.py first, then ensure this path is correct.")
    else:
        try:
            model, processor, device = load_model_and_processor(model_path_to_test)
            emotion_labels_list = load_emotion_labels(model_path_to_test)

            if not emotion_labels_list:
                print("No emotion labels loaded, cannot proceed with creating DataFrame for emotions.")
            else:
                # Load audio using librosa
                y, sr = librosa.load(dummy_audio_path, sr=16000) # Ensure 16kHz for Whisper
                
                full_transcription, segment_emotion_probabilities = perform_inference(
                    y, sr, model, processor, device, segment_duration=5
                )
                
                print("\n--- Example Inference Results ---")
                print(f"Full Transcription: {full_transcription}")
                
                for i, probs in enumerate(segment_emotion_probabilities):
                    if probs is not None:
                        # Create DataFrame for display
                        df = pd.DataFrame({'Emotion': emotion_labels_list, 'Probability': probs})
                        df = df.sort_values(by='Probability', ascending=False)
                        print(f"\nSegment {i+1} Emotion Probabilities:")
                        print(df)
                    else:
                        print(f"\nSegment {i+1}: No emotion probabilities.")
        except FileNotFoundError as e:
            print(f"Skipping example run due to FileNotFoundError: {e}")
        except Exception as e:
            print(f"An error occurred during the example run: {e}")
            import traceback
            traceback.print_exc()

    print("inference.py example finished.")
