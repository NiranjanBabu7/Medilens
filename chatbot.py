# src/chatbot.py
import os
import dotenv
from .cyborg_client import CyborgClient
from .embed import Embedder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import List

dotenv.load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-small")

class MediLensChatbot:
    def __init__(self, llm_model=LLM_MODEL):
        print(f"[MediLensChatbot] Loading LLM: {llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
        self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        self.embedder = Embedder()
        self.cyborg = CyborgClient()
        self.index = None

    def load_index(self):
        self.index = self.cyborg.create_encrypted_index()
        return self.index

    def query_and_answer(self, query_text: str, k=5) -> dict:
        # 1. embed query
        q_emb = self.embedder.model.encode([query_text])[0].tolist()
        # 2. ensure index loaded
        if self.index is None:
            self.load_index()
        # 3. query
        res = self.cyborg.query_index(self.index, q_emb, k=k)
        hits = res["results"]
        # 4. build context — fuse retrieved content in a prompt
        context_pieces = []
        for h in hits:
            # the SDK result format may be: {'id':..., 'score':..., 'metadata':..., 'content':...}
            # adapt as needed
            content = h.get("content") if isinstance(h, dict) else h
            if content:
                context_pieces.append(f"- {content[:600]}")
        context = "\n".join(context_pieces) if context_pieces else "No contextual records found."
        prompt = f"""You are a clinical-assistant. Using the retrieved anonymized patient context below, provide a safe, evidence-based, and conservative clinical answer. Do NOT reveal PHI. If insufficient data, say so.

Context:
{context}

Question:
{query_text}

Answer:"""
        # 5. call LLM to generate an answer
        gen = self.generator(prompt, max_length=512, do_sample=False)[0]["generated_text"]
        # 6. return answer + provenance (ids + scores) for audit
        return {
            "query": query_text,
            "answer": gen,
            "retrieval_latency": res["latency"],
            "retrieved": hits
        }
