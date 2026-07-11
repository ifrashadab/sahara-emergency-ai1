# main.py
"""
SAHARA AI Triage Service — RAG + LLM edition.

Replaces the old classifier -> static-dictionary-lookup flow with:
  1. Retrieval: embed the user's message, pull the most relevant first-aid /
     safety guidance chunks from a Chroma vector store.
  2. Generation: send the user's message + retrieved context to an LLM
     (Claude), which reasons over the retrieved guidance instead of just
     regurgitating a fixed template.
  3. Agentic escalation: the LLM has one tool, `flag_for_escalation`, which
     it calls when it judges the situation to be medium/high/critical
     severity. This is what turns the service from "answer a question" into
     "make a judgment call and act on it."

Run:
    uvicorn main:app --reload --port 8000

Requires ANTHROPIC_API_KEY to be set (see .env.example).
"""

import os
import json
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
import anthropic

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "sahara_first_aid_kb"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"
TOP_K = 3

app = FastAPI(title="SAHARA AI Triage — RAG Edition")

# CORS: allow the SAHARA frontend (served from anywhere during dev / your
# deployed domain in prod) to call this API directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Connecting to vector store...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL_NAME
)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME, embedding_function=embed_fn
)
print("Vector store ready.")

llm_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


class TriageRequest(BaseModel):
    text: str
    help_type: Optional[str] = None  # e.g. "Medical", "Safety/Escort" — from the dropdown in askhelp.html


class EscalationFlag(BaseModel):
    severity: str
    reason: str


# --- Tool definition: the one action the model is allowed to take ---------
ESCALATION_TOOL = {
    "name": "flag_for_escalation",
    "description": (
        "Call this when the situation described is medium, high, or critical "
        "severity and should be flagged for urgent human/volunteer attention "
        "rather than just receiving informational guidance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "severity": {
                "type": "string",
                "enum": ["medium", "high", "critical"],
                "description": "How urgent this situation is.",
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining why this severity was chosen.",
            },
        },
        "required": ["severity", "reason"],
    },
}


def retrieve_context(query: str, k: int = TOP_K):
    results = collection.query(query_texts=[query], n_results=k)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(
            {
                "condition": meta["condition"],
                "category": meta["category"],
                "source": meta["source"],
                "content": doc,
            }
        )
    return chunks


def build_system_prompt(context_chunks) -> str:
    context_text = "\n\n".join(
        f"[{c['condition']} | {c['category']} | source: {c['source']}]\n{c['content']}"
        for c in context_chunks
    )
    return f"""You are SAHARA's emergency guidance assistant. A user in a
potentially urgent situation has described what's happening. You have
retrieved the following knowledge base entries that may be relevant:

{context_text}

Instructions:
- Ground your response in the retrieved entries above. If none of them
  genuinely match the situation, say so and give general safety guidance
  instead of forcing a mismatched entry.
- Give clear, numbered, actionable steps — this may be read by someone
  scared or in a hurry.
- Always state plainly if the person should contact emergency services.
- If the situation is medium, high, or critical severity, call the
  flag_for_escalation tool with your severity judgment before or alongside
  your written answer.
- You are not a substitute for professional medical, mental health, or law
  enforcement response — say so briefly when relevant, without being so
  verbose it buries the actionable steps.
- Keep your written answer focused: a short severity read, then steps."""


@app.post("/triage")
def triage(request: TriageRequest):
    context_chunks = retrieve_context(request.text)

    system_prompt = build_system_prompt(context_chunks)
    user_message = request.text
    if request.help_type:
        user_message = f"[Reported help type: {request.help_type}]\n{user_message}"

    response = llm_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        system=system_prompt,
        tools=[ESCALATION_TOOL],
        messages=[{"role": "user", "content": user_message}],
    )

    escalation: Optional[EscalationFlag] = None
    answer_text = ""

    for block in response.content:
        if block.type == "text":
            answer_text += block.text
        elif block.type == "tool_use" and block.name == "flag_for_escalation":
            escalation = EscalationFlag(
                severity=block.input.get("severity", "medium"),
                reason=block.input.get("reason", ""),
            )

    return {
        "answer": answer_text.strip(),
        "escalate": escalation is not None,
        "severity": escalation.severity if escalation else "low",
        "escalation_reason": escalation.reason if escalation else None,
        "retrieved_sources": [
            {"condition": c["condition"], "source": c["source"]}
            for c in context_chunks
        ],
    }


@app.get("/")
def read_root():
    return {"message": "SAHARA RAG Triage Server is running."}
