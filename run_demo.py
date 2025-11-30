# src/run_demo.py
import sys
from datetime import datetime
from .embed import build_embeddings
from .cyborg_client import CyborgClient

# Sample EHR data
SAMPLE_RECORDS = [
    {"anon_id": "patient_001", "text_masked": "Patient has mild fever and headache.", "timestamp": str(datetime.now())},
    {"anon_id": "patient_002", "text_masked": "Patient reports shortness of breath and cough.", "timestamp": str(datetime.now())},
    {"anon_id": "patient_003", "text_masked": "Patient shows signs of elevated blood pressure.", "timestamp": str(datetime.now())},
]

def ingest():
    print("Preprocessing data (masking PHI)...")
    # Normally, you would load data from file and mask PHI
    masked_records = SAMPLE_RECORDS  

    print("Building embeddings...")
    vectors = build_embeddings(masked_records)

    print("Upserting vectors to CyborgDB...")
    cy = CyborgClient()
    index = cy.create_encrypted_index()
    cy.upsert_vectors(index, vectors)

    print("Ingest complete.")
    return cy, index

def query_demo(cy, index):
    # Example query: just use first record's vector
    query_vector = index.data[0]["vector"]
    print("\n[Query Demo] Top results:")
    res = cy.query_index(index, query_vector, k=3)
    for i, r in enumerate(res["results"], 1):
        print(f"{i}. ID: {r['id']}, Content: {r['content']}, Timestamp: {r['metadata']['timestamp']}")
    print(f"\nQuery latency: {res['latency']:.4f} seconds")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        cy, index = ingest()
        query_demo(cy, index)
    else:
        print("Usage: python run_demo.py ingest")
