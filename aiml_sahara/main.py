# main.py — SAHARA AI Triage Service (RAG + LLM, Groq free-tier edition)
#
# Same architecture as before, but the LLM is now Llama 3.3 70B served by
# Groq's free API instead of Anthropic Claude. The /triage response format is
# unchanged, so the frontend (assistant.html / askhelp.html) needs no edits.
#
# Run:
#   pip3 install groq --user           (one extra dependency)
#   put GROQ_API_KEY=gsk_... in .env
#   python3 -m uvicorn main:app --reload --port 8000

import os
import json
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "sahara_first_aid_kb"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 3

app = FastAPI(title="SAHARA AI Triage — RAG (Groq)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)
from case_api import router as case_router
app.include_router(case_router)

print("Connecting to vector store...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(
    name=COLLECTION_NAME, embedding_function=embed_fn
)
print("Vector store ready.")

llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class TriageRequest(BaseModel):
    text: str
    help_type: Optional[str] = None


# --- Tool definition (OpenAI/Groq function-calling format) -----------------
ESCALATION_TOOL = {
    "type": "function",
    "function": {
        "name": "flag_for_escalation",
        "description": (
            "Call this when the situation described is medium, high, or "
            "critical severity and should be flagged for urgent "
            "human/volunteer attention rather than just receiving "
            "informational guidance."
        ),
        "parameters": {
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
  flag_for_escalation tool with your severity judgment. Also always write
  your guidance text — never respond with only a tool call.
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = llm_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=800,
        messages=messages,
        tools=[ESCALATION_TOOL],
        tool_choice="auto",
    )

    choice = response.choices[0].message
    answer_text = choice.content or ""
    escalation = None

    if choice.tool_calls:
        for tc in choice.tool_calls:
            if tc.function.name == "flag_for_escalation":
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                escalation = {
                    "severity": args.get("severity", "medium"),
                    "reason": args.get("reason", ""),
                }

    # Some models return ONLY a tool call with empty text. If so, make a
    # second call without tools to get the written guidance.
    if not answer_text.strip():
        followup = llm_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=800,
            messages=messages
            + [
                {
                    "role": "assistant",
                    "content": "(Severity has been assessed. Now writing the guidance.)",
                },
                {
                    "role": "user",
                    "content": "Now write the step-by-step guidance for the situation above.",
                },
            ],
        )
        answer_text = followup.choices[0].message.content or ""

    return {
        "answer": answer_text.strip(),
        "escalate": escalation is not None,
        "severity": escalation["severity"] if escalation else "low",
        "escalation_reason": escalation["reason"] if escalation else None,
        "retrieved_sources": [
            {"condition": c["condition"], "source": c["source"]}
            for c in context_chunks
        ],
    }


@app.get("/")
def read_root():
    return {"message": "SAHARA RAG Triage Server is running."}
