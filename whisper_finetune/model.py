import torch 
import torch.nn as nn 
from transformers import WhisperPreTrainedModel, WhisperModel, WhisperConfig 
from typing import Optional, Tuple, List, Dict, Any, Union 

class EmotionWhisperModel(WhisperPreTrainedModel):
    """Whisper + emotion classification head"""
    def __init__(self, config: WhisperConfig, num_emotions_classes: int = 26):
        super().__init__(config)
        
        #load whisper model components 
        self.whisper = WhisperModel(config) #imports base encoder-decoder architecture without output projections 
        
        #original projection for transcription 
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        #projection for emotion classification 
        self.emotion_classifier = nn.Linear(config.d_model, num_emotions_classes)
        
        #initialise weights 
        self.post_init()
        
    def _get_segment_representations(self, decoder_output, timestamp_indices):
        """
        Extract segment-level representations from decoder output 
        """
        batch_size = decoder_output.size(0)
        segment_reps = []
        
        for b in range(batch_size):
            ts_indices = timestamp_indices[b] #each timestamp in ts_indices is a list of end indices for a segment 
            segments = []
            start_idx = 0 #first segment starts at 0 
            
            #process each segment defined by timestamps 
            for end_idx in ts_indices: 
                if end_idx > start_idx:
                    # segment tokens selects tokens from start_idx to end_idx for batch item b 
                    segment_tokens = decoder_output[b, start_idx:end_idx] #shape: [segment_length, d_model]
                    
                    seg_rep = torch.mean(segment_tokens, dim=0) #average pooling over segment tokens 
                    
                    segments.append(seg_rep)
                start_idx = end_idx + 1 #move to next segment 
            
            #handle the last segment 
            if start_idx < decoder_output.size(1): #check if there are remaining tokens
                last_segment_tokens = decoder_output[b, start_idx:]
                seg_rep = torch.mean(last_segment_tokens, dim=0)
                segments.append(seg_rep)
                
            #if no semgents, use global pooling 
            if not segments:
                segments = torch.mean(decoder_output[b], dim=0)
            
            segment_reps.append(torch.stack(segments))
        
        return segment_reps 
    
    def forward(self, 
                input_features, 
                attention_mask=None, 
                decoder_input_ids=None, 
                decoder_attention_mask=None, 
                labels=None, 
                emotion_labels=None, 
                timestamp_indices=None, 
                return_dict=True):
        """forward pass with transcription + emotion prediction"""
        
        #run base whisper model 
        outputs = self.whisper(
            input_features, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids, 
            decoder_attention_mask=decoder_attention_mask, 
            return_dict=return_dict, 
        )
        
        # get decoder outputs 
        hidden_states = outputs.last_hidden_state 
        
        # projection for transcription (token-level)
        logits = self.proj_out(hidden_states)
        
        #extract segment representations and predict emotions 
        emotion_logits = None 
        if timestamp_indices is not None:
            #segment level emotion classification (for inference)
            segment_reps = self._get_segment_representations(hidden_states, timestamp_indices)
            
            #predict emotions for each segment 
            emotion_logits = []
            for seq_reps in segment_reps: 
                seq_emotion_logits = self.emotion_classifier(seq_reps)
                emotion_logits.append(seq_emotion_logits)
        else:
            #sequence level emotion classification (for training)
            #global pooling over sequence length 
            global_representation = torch.mean(hidden_states, dim=1) #[batch_size, d_model]
            emotion_logits = self.emotion_classifier(global_representation) #[batch_size, num_emotions_classes]
                
        #prepare output dictionary 
        result = {
            "logits": logits, 
            "emotion_logits": emotion_logits, 
            # "hidden_states": outputs.hidden_states, 
            # "attentions": outputs.attentions, 
        }
        
        return result 
       

def load_emotion_whisper_model(num_emotions_classes=26):
    """Load pretrained whisper medium model and adds emotion classification head"""
    
    #load pretrained model and config 
    from transformers import WhisperConfig, WhisperProcessor #we need whisper proecessor for tokenization and feature extraction 
    
    #load pretrained whisper medium 
    model_id = "openai/whisper-tiny"
    config = WhisperConfig.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    
    #intiialise our emotion-aware model with the config 
    model = EmotionWhisperModel(config, num_emotions_classes=num_emotions_classes)
    
    #load pretrained weights for the whisper components 
    pretrained_model = WhisperModel.from_pretrained(model_id)
    model.whisper.load_state_dict(pretrained_model.state_dict())
    
    #initialise projection layer with the pretrained weights if available 
    if hasattr(pretrained_model, "proj_out"):
        model.proj_out.load_state_dict(pretrained_model.proj_out.state_dict())
    
    return model, processor 