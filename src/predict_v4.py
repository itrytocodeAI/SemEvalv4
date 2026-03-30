"""
predict_v4.py
Inference loop for DimABSA V4.
"""

import json
import torch
import numpy as np
import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_v4 import MODEL_NAME, CHECKPOINT_DIR, BATCH_SIZE, MAX_LEN, va_denormalise, PREDICTION_DIR, DATA_ROOT, CATEGORY2IDX, GRID_POSITIVE
from src.data_loader_v4 import get_dataloader
from src.model_v4 import DimABSAV4Model

@torch.no_grad()
def run_predict(subtask: int, lang: str="eng", domain: str="restaurant", seed: int=42):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"st{subtask}_{lang}_{domain}_seed{seed}"
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return
        
    model = DimABSAV4Model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    st_path = DATA_ROOT / f"subtask_{subtask}" / lang
    dev_key_part = "task1" if subtask == 1 else "task2" if subtask == 2 else "task3"
    dev_file = st_path / f"{lang}_{domain}_dev_{dev_key_part}.jsonl"
    
    if not dev_file.exists():
        print(f"Data file {dev_file} missing.")
        return
        
    dl = get_dataloader(dev_file, subtask, tokenizer, BATCH_SIZE, MAX_LEN, is_train=False)
    
    idx2cat = {v:k for k,v in CATEGORY2IDX.items()}
    
    results = []
    
    # We load raw dev file to match ids/texts
    orig_data = []
    with open(dev_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                orig_data.append(json.loads(line))
                
    curr_orig_idx = 0
    # Process
    if subtask == 1:
        # It's flattened in dataloader (text x aspect)
        # So we accumulate
        out_dict = collections.defaultdict(list)
        batch_idx = 0
        
        for batch in dl:
            pred_va = model.forward_regression(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
                batch["ling_features"].to(device),
                batch["aspect_token_mask"].to(device)
            )
            v_preds = pred_va[:, 0].float().cpu().numpy()
            a_preds = pred_va[:, 1].float().cpu().numpy()
            
            # The dataloader iterates over instances exactly in the order of flattened array.
            # We must map them back to their original IDs.
            # The dataloader yields instances... 
            # In data_loader_v4, instances are list of {id, text, aspect, etc}
            batch_size_actual = v_preds.shape[0]
            for i in range(batch_size_actual):
                inst = dl.dataset.instances[batch_idx]
                id_ = inst["id"]
                aspect = inst["aspect"]
                v = va_denormalise(v_preds[i])
                a = va_denormalise(a_preds[i])
                out_dict[id_].append({"Aspect": aspect, "VA": f"{v:.2f}#{a:.2f}"})
                batch_idx += 1
                
        for obj in orig_data:
            id_ = obj["ID"]
            text = obj["Text"]
            results.append({"ID": id_, "Text": text, "Aspect_VA": out_dict[id_]})
            
    else:
        print(f"Prediction for extraction tasks (2/3) not fully mapped yet.")
        # Phase 4 would complete extraction token decoding logic.
        return
        
    out_path = PREDICTION_DIR / f"{run_name}_predictions.jsonl"
    with open(out_path, "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Saved ST{subtask} predictions to {out_path}")

if __name__ == "__main__":
    import collections
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", type=int, default=1)
    parser.add_argument("--lang", type=str, default="eng")
    parser.add_argument("--domain", type=str, default="restaurant")
    args = parser.parse_args()
    run_predict(args.subtask, args.lang, args.domain)
