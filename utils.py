# src/utils.py
import re
import json
from typing import Dict

def simple_phi_mask(text: str) -> str:
    """
    Basic PHI masking:
    - Remove MRN-like numbers
    - Remove phone numbers, dates (YYYY or YYYY-MM-DD), and emails
    - Replace names that appear in PATIENT: with 'REDACTED'
    NOTE: This is intentionally simplistic; use certified de-id for production.
    """
    masked = text
    # emails
    masked = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', '[REDACTED_EMAIL]', masked)
    # phone numbers
    masked = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', masked)
    # dates like YYYY-MM-DD or MM/DD/YYYY
    masked = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[REDACTED_DATE]', masked)
    masked = re.sub(r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b', '[REDACTED_DATE]', masked)
    # MRN-type numbers (long digits)
    masked = re.sub(r'\b\d{6,}\b', '[REDACTED_ID]', masked)
    # simple name tags
    masked = re.sub(r'(?i)(patient|name):\s*[A-Z][a-z]+', r'\1: [REDACTED_NAME]', masked)
    return masked

def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(path: str, items):
    import json
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
