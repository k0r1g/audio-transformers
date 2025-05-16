import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tempfile # For handling uploaded files safely
import os

# Import functions from inference.py
from inference import load_model_and_processor, load_emotion_labels, perform_inference, DEFAULT_MODEL_PATH

# --- Model and Processor Loading (Cached) ---
@st.cache_resource # Caches the model and processor resource
def cached_load_model_and_processor(model_path = DEFAULT_MODEL_PATH):
    try:
        model, processor, device = load_model_and_processor(model_path)
        return model, processor, device
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}. Please ensure the model path is correct and the model files are present at '{model_path}'. You might need to train the model first or adjust the DEFAULT_MODEL_PATH in inference.py.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None, None, None

@st.cache_data # Caches the emotion labels data
def cached_load_emotion_labels(model_path = DEFAULT_MODEL_PATH):
    labels = load_emotion_labels(model_path)
    # The load_emotion_labels function in inference.py now handles fallbacks internally
    # and will print warnings to the console if fallbacks are used.
    if not labels: # Should ideally not happen if fallback is always a list
        st.error("Critical error: Emotion labels list is unexpectedly empty even after fallback logic. Check inference.py.")
        return ["error"] # A generic error marker if something went very wrong
    if len(labels) != 10: # Add a check for the expected number of labels
        st.warning(f"Warning: Loaded emotion labels count ({len(labels)}) is not 10. Display might be affected. Labels: {labels}")
    return labels

# --- Main Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="Emotion-Aware Audio Transcription")
    st.title("🎤 Emotion-Aware Audio Transcription & Analysis")

    # --- Load Model, Processor, and Labels ---
    model_load_path = DEFAULT_MODEL_PATH 
    
    model, processor, device = cached_load_model_and_processor(model_load_path)
    emotion_labels_list = cached_load_emotion_labels(model_load_path)

    if model is None or processor is None or device is None:
        st.error("Model, processor, or device could not be loaded. Please check the console for errors. The application cannot proceed.")
        st.stop()
    
    # Inform user if fallback might have occurred (based on console logs from inference.py)
    # The list itself will be populated by fallback if style_to_id.txt failed.
    # We can check its content if we want to be more explicit in UI, but for now relying on console.

    # --- UI for File Upload ---
    st.sidebar.header("Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Choose an audio file...", type=["wav", "mp3", "flac", "ogg", "m4a"])

    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        st.audio(uploaded_file, format=uploaded_file.type)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_audio_file:
                tmp_audio_file.write(uploaded_file.getvalue())
                tmp_audio_file_path = tmp_audio_file.name
            
            audio_waveform, sampling_rate = librosa.load(tmp_audio_file_path, sr=16000)
            duration = librosa.get_duration(y=audio_waveform, sr=sampling_rate)
            st.sidebar.info(f"Audio Duration: {duration:.2f} seconds")

            if st.button("Transcribe and Analyze Emotions"):
                with st.spinner("Processing audio... This might take a moment."):
                    full_transcription, segment_emotion_probabilities = perform_inference(
                        audio_waveform, sampling_rate, model, processor, device, segment_duration=5
                    )
                
                st.markdown("---")
                st.subheader("Transcription:")
                if full_transcription:
                    st.markdown(f"> {full_transcription}")
                else:
                    st.warning("Transcription could not be generated.")
                st.markdown("---")

                st.subheader("Emotion Analysis (per 5-second segment):")
                
                if not segment_emotion_probabilities:
                    st.warning("No emotion analysis results were generated.")
                else:
                    for i, probs_array in enumerate(segment_emotion_probabilities):
                        if probs_array is None:
                            st.markdown(f"**Segment {i+1}:** Error during processing.")
                            continue

                        start_time = i * 5
                        end_time = min((i + 1) * 5, duration)
                        st.markdown(f"**Segment {i+1} ({start_time:.2f}s - {end_time:.2f}s):**")
                        
                        # Check if the number of probabilities matches the number of labels
                        if len(probs_array) == len(emotion_labels_list):
                            df_emotions = pd.DataFrame({
                                'Emotion': emotion_labels_list,
                                'Probability': probs_array
                            })
                            st.bar_chart(df_emotions.set_index('Emotion'))
                        else:
                            st.text(f"Raw Probabilities (label/probability count mismatch): {probs_array}")
                            st.text(f"Expected {len(emotion_labels_list)} labels for {len(probs_array)} probabilities.")
                        st.markdown("---")
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            import traceback
            st.exception(traceback.format_exc())
        finally:
            if 'tmp_audio_file_path' in locals() and os.path.exists(tmp_audio_file_path):
                os.remove(tmp_audio_file_path)
    else:
        st.info("Upload an audio file using the sidebar to get started.")

if __name__ == "__main__":
    main() 