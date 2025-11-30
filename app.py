# src/app.py
import _disable_tf
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import streamlit as st
from datetime import datetime

from cyborg_client import CyborgClient
from embed import Embedder, build_embeddings

# Streamlit settings
st.set_page_config(page_title="MediLens Demo", layout="wide")
st.title("🩺 MediLens: AI-powered Medical Vector Search")

# -----------------------------
# Initialize Embedder (singleton)
embedder = Embedder()

# Initialize CyborgClient
cyborg_client = CyborgClient()

# -----------------------------
# ✅ Option 1: Ensure fresh index
INDEX_NAME = "medi-lens-index"

# Delete existing index if it exists
try:
    cyborg_client.delete_index(INDEX_NAME)
    st.info(f"Existing index '{INDEX_NAME}' deleted for fresh ingestion.")
except Exception:
    pass  # Ignore if index doesn't exist

# Create a new index with the same embedding dimension as Embedder
index = cyborg_client.create_encrypted_index(INDEX_NAME)
st.success(f"Fresh index '{INDEX_NAME}' created successfully.")
# -----------------------------

# -------------------------------------------------------------------
# 📌 SECTION 1 — INGEST PATIENT DATA
# -------------------------------------------------------------------
st.header("📥 Ingest Patient Data")

with st.form("ingest_form"):
    patient_id = st.text_input("Patient ID", placeholder="patient_004")
    patient_text = st.text_area("Patient Notes", placeholder="Describe patient symptoms, diagnosis, or findings...")

    submitted = st.form_submit_button("Ingest Data")

    if submitted:
        if not patient_id or not patient_text:
            st.error("Please enter both Patient ID and Notes.")
        else:
            record = [{
                "anon_id": patient_id,
                "text_masked": patient_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]

            try:
                # Generate embeddings with consistent model
                vectors = build_embeddings(record, embedder)
                # Upsert into CyborgDB
                cyborg_client.upsert_vectors(index, vectors)
                st.success(f"Record for **{patient_id}** ingested successfully!")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

# -------------------------------------------------------------------
# 📌 SECTION 2 — QUERY PATIENT RECORDS
# -------------------------------------------------------------------
st.header("🔍 Query Patient Records")

query_text = st.text_input("Enter your query", placeholder="e.g., fever, headache, high BP")

if st.button("Search Records"):
    if not query_text:
        st.error("Please enter a search query.")
    else:
        try:
            # Generate query vector using same model
            query_vector = embedder.embed_texts([query_text])[0]

            # Query CyborgDB
            results = cyborg_client.query_index(index, query_vector, k=5).get("results", [])

            st.subheader("Top Results")
            if not results:
                st.info("No matching records found.")
            else:
                for i, r in enumerate(results, 1):
                    st.write(
                        f"""
                        **{i}. Patient ID:** {r.get('id')}  
                        **Notes:** {r.get('content')}  
                        **Timestamp:** {r.get('metadata', {}).get('timestamp', 'N/A')}
                        """
                    )

        except Exception as e:
            st.error(f"Search failed: {e}")

