# src/data_preprocess.py
import os
from .utils import simple_phi_mask, read_jsonl, write_jsonl
from typing import List
import argparse

def preprocess_file(inpath: str, outpath: str):
    items = []
    for rec in read_jsonl(inpath):
        rec_copy = rec.copy()
        rec_copy['text_masked'] = simple_phi_mask(rec.get('text',''))
        # You may also remove patient_id entirely and replace with anonymized id
        rec_copy['anon_id'] = rec.get('patient_id', 'anon')  # already anonymized in sample
        items.append(rec_copy)
    write_jsonl(outpath, items)
    print(f"Wrote {len(items)} records to {outpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inpath", default="data/sample_ehr.jsonl")
    parser.add_argument("--out", dest="outpath", default="data/sample_ehr_masked.jsonl")
    args = parser.parse_args()
    preprocess_file(args.inpath, args.outpath)
