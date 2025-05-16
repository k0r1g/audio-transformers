#!/usr/bin/env python
# evaluate_fullseq.py
# -------------------------------------------------------------

import os, argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from transformers import WhisperProcessor, GenerationConfig
from model        import load_emotion_whisper_model          # <—
from dataset      import create_dataset, SIMPLE_STYLES


# ----------------------- CLI --------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Sequence-level emotion evaluation (no segments)")
    p.add_argument("--model_path",
                   default="./emotion_whisper_model/best_model_epoch7",
                   help="Directory with fine-tuned weights OR a HF hub id")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--simple_styles", action="store_true",
                   help="Restrict to the 10 SIMPLE_STYLES")
    p.add_argument("--output_dir",   default="./eval_out")
    return p.parse_args()


# --------------------- main ---------------------------------
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Device:", device)

    # ---------- processor ------------------------------------
    print(f"Loading model from: {args.model_path}")
    processor = WhisperProcessor.from_pretrained(args.model_path)

    # ---------- model architecture + weights -----------------
    print("Attempting to load model weights...")
    from model import EmotionWhisperModel  # Import the specific model class
    
    # Load the model - from_pretrained will try local path first, then Hub
    model = EmotionWhisperModel.from_pretrained(args.model_path)
    print(f"Successfully loaded model from: {args.model_path}")
    
    # ------------------------------------------------------------------
    #  Tie the output projection to the shared decoder embeddings
    # ------------------------------------------------------------------
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
    
    model.to(device).eval()

    # ---------- generation config ----------------------------
    try:
        gen_cfg = GenerationConfig.from_pretrained(args.model_path)
        print("✓ generation_config.json found with the model")
    except (OSError, ValueError):
        gen_cfg = GenerationConfig.from_pretrained("openai/whisper-tiny")
        print("➜ No generation_config.json found with the model – borrowed one from "
              "openai/whisper-tiny")

    # --- kill all stored forced-id copies ---------------------
    gen_cfg.forced_decoder_ids = None           # <-- NEW

    if hasattr(model, "whisper") and model.whisper is not None:
        model.whisper.config.forced_decoder_ids = None   # <-- NEW
        model.whisper.generation_config = gen_cfg        # re-attach
    else:
        print("Warning: model.whisper attribute not found or is None. Skipping whisper-specific config updates.")


    # ---------------- dataset --------------------------------
    styles         = SIMPLE_STYLES if args.simple_styles else None
    # Pass processor correctly
    _, _, test_ds, style2idx = create_dataset(processor=processor, selected_styles=styles) 
    idx2style = {v: k for k, v in style2idx.items()}

    # Print the emotion classes being used
    current_emotion_classes = sorted(list(style2idx.keys()))
    print(f"Evaluating with the following {len(current_emotion_classes)} emotion classes:")
    for emotion_class in current_emotion_classes:
        print(f"- {emotion_class}")
    print("---")

    loader = DataLoader(test_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=test_ds.collate_fn)

    # --------------- evaluation ------------------------------
    pad_id = processor.tokenizer.pad_token_id
    gt_all, pred_all = [], []

    log_file_path = Path(args.output_dir) / "predictions.txt"
    print(f"Logging predictions to: {log_file_path}")
    with open(log_file_path, "w") as log_file:

        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            feats   = batch["input_features"].to(device)      # (B, 80, T)
            if feats.shape[1] != 80:                          # safety for (B,T,80)
                feats = feats.transpose(1, 2)

            gt_tok  = batch["labels"]
            gt_emo  = batch["emotion_labels"]

            with torch.no_grad():
                # 1) make Whisper spit out an English transcription
                if hasattr(model, 'whisper') and model.whisper is not None:
                    gen_ids = model.whisper.generate(
                        feats,
                        max_new_tokens=100,           # Further reduced
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.eos_token_id, 
                        do_sample=False,              
                        no_repeat_ngram_size=3,       
                        repetition_penalty=1.15,      
                        length_penalty=-0.5,          # Encourage shorter output
                        forced_decoder_ids=None     
                    )
                else:
                    print("Error: model.whisper not found. Cannot generate text.")
                    continue # Skip this batch
                
                # 2) run our classification head once per sample
                outs = model(input_features=feats,
                             decoder_input_ids=gen_ids)       # full-sequence
                pred_emo = outs["emotion_logits"].argmax(dim=-1)

            # ------------- print / log ----------------------------
            for i in range(len(gen_ids)):
                gt_e   = int(gt_emo[i])
                pr_e   = int(pred_emo[i])

                gt_txt = processor.decode(
                    [t for t in gt_tok[i] if t != pad_id],
                    skip_special_tokens=True).strip()
                pr_txt = processor.decode(gen_ids[i], skip_special_tokens=True)

                print(f"GT  emotion: {idx2style.get(gt_e, 'Unknown')}\n"
                      f"Pred emotion: {idx2style.get(pr_e, 'Unknown')}\n"
                      f"GT  text   : {gt_txt}\n"
                      f"Pred text : {pr_txt}\n{'-'*40}")

                log_file.write(f"{idx2style.get(gt_e, 'Unknown')}\t{idx2style.get(pr_e, 'Unknown')}\t"
                               f"{gt_txt}\t{pr_txt}\n")

                gt_all.append(gt_e)
                pred_all.append(pr_e)

    # ---------------- metrics --------------------------------
    acc = accuracy_score(gt_all, pred_all) if gt_all else 0.0
    f1  = f1_score(gt_all, pred_all, average="weighted",
                   zero_division=0) if gt_all else 0.0

    print("\n===== SUMMARY =====")
    print("Samples    :", len(gt_all))
    print("Accuracy   :", round(acc, 4))
    print("Weighted F1:", round(f1, 4))

    metrics_file_path = Path(args.output_dir) / "metrics.txt"
    print(f"Saving metrics to: {metrics_file_path}")
    with open(metrics_file_path, "w") as f:
        f.write(f"samples\t{len(gt_all)}\n"
                f"accuracy\t{acc:.6f}\n"
                f"weighted_f1\t{f1:.6f}\n")

    print(f"Evaluation complete. Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()
