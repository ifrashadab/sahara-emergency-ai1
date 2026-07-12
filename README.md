# SAHARA AI — Community Emergency Response Platform

> **Someone nearby, always.** SAHARA connects people in distress with AI-guided
> first-aid & safety support and real, location-matched volunteers — in seconds.

## Problem statement

In an emergency, the gap between "something is wrong" and "help has arrived" is
where outcomes are decided. Formal emergency services are essential but slow to
mobilize for many situations — and entire categories of distress (harassment,
feeling followed, panic attacks, being stranded at night) don't map cleanly to
a 112 call at all. SAHARA fills that gap with two things working together:

1. **Grounded AI guidance** — immediate, step-by-step instructions retrieved
   from verified first-aid and safety sources (not hallucinated by a chatbot).
2. **A community volunteer network** — verified nearby volunteers who are
   alerted with the requester's location and situation when the AI judges the
   case urgent.

## Architecture

```
                        ┌──────────────────────────────┐
                        │        Frontend (HTML/JS)     │
                        │  landing.html  assistant.html │
                        │  askhelp.html  dashboards …   │
                        └──────┬───────────────┬───────┘
                               │               │
                 POST /triage  │               │  Firestore SDK
                               ▼               ▼
                 ┌──────────────────┐   ┌─────────────────┐
                 │  FastAPI backend │   │    Firebase      │
                 │    (main.py)     │   │ help_requests,   │
                 └───┬──────────┬───┘   │ chat, volunteers │
                     │          │       └─────────────────┘
        1. embed     │          │  4. LLM + severity tool
           query     ▼          ▼
             ┌────────────┐  ┌──────────────────┐
             │  ChromaDB  │  │  Claude (LLM)     │
             │ vector     │  │  tools:           │
             │ store      │  │  flag_for_        │
             │ (KB chunks)│  │  escalation()     │
             └────────────┘  └──────────────────┘
                     ▲
                     │  ingest.py (rebuild on KB edit)
             knowledge_base_docs.json
             (NHS / Red Cross / St John Ambulance–based entries)
```

## The AI pipeline (RAG workflow)

Every `/triage` request runs six steps:

1. **Embed** the user's message locally with `sentence-transformers`
   (all-MiniLM-L6-v2 — free, no API dependency for retrieval).
2. **Retrieve** the top-3 semantically closest entries from a persistent
   ChromaDB collection covering 15 emergency categories: medical (heart
   attack, stroke, choking, burns, bleeding, seizures, anaphylaxis, drowning,
   fainting, asthma), mental health (panic attacks, suicidal crisis),
   harassment/assault support, safety-escort, and lost/stranded.
3. **Ground** the LLM: the retrieved entries are injected into the system
   prompt with an explicit instruction to answer *only* from them — and to say
   so when nothing relevant was found instead of improvising.
4. **Judge severity**: the model has exactly one tool, `flag_for_escalation
   (severity, reason)`, which it calls for medium/high/critical cases. This
   tool call is what turns the service from a Q&A bot into an agent that makes
   a decision.
5. **Respond** with the grounded answer, severity, escalation flag, and the
   sources used.
6. **Explain**: the frontend renders an expandable "How I got this answer"
   panel showing the pipeline and retrieved chunks — explainability is a
   first-class UI feature, not a debug view.

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | Vanilla HTML/CSS/JS | Zero build step, instant load, hosts anywhere |
| Realtime data | Firebase (Firestore + RTDB) | Help requests, chat, volunteer presence |
| API | FastAPI (Python) | Async, typed, easy to deploy |
| Vector store | ChromaDB (persistent, local) | Free, no external dependency for retrieval |
| Embeddings | sentence-transformers (MiniLM) | Local, no per-query embedding cost |
| LLM | Anthropic Claude + tool use | Grounded generation + the escalation tool call |

## Key design decisions (interview notes)

- **Retrieval over classification.** The original prototype used a fine-tuned
  text classifier with an exact-match dictionary lookup — "can't breathe"
  failed to match "Bronchial Asthma" and fell through to a generic default.
  Semantic retrieval solves the paraphrase problem and lets the knowledge base
  grow without retraining anything.
- **One tool, deliberately.** The model can take exactly one action: flag for
  escalation. Constraining the action space keeps a safety-critical system
  auditable — every escalation carries a machine-readable severity and reason.
- **Fail-open for safety.** If the LLM API is unreachable, the UI degrades to
  directing users to SOS/emergency services and the human volunteer flow —
  the AI layer augments the human network, it never gates it.
- **Red means one thing.** In the design system, the color red appears in
  exactly one place: emergency actions (SOS). In a panic, findability beats
  aesthetics.
- **Explainability as UX.** Showing retrieved sources isn't just for
  engineers — it's a trust feature for users receiving safety-critical advice.

## Folder structure

```
sahara-codex/
├── frontend/
│   ├── landing.html        # redesigned landing page
│   ├── assistant.html      # AI chat (conversations, severity, explainability)
│   ├── askhelp.html        # SOS / help-request form (Firestore)
│   ├── dashboard.html, livevolunteer.html, leaderboard.html, …
├── aiml_sahara/
│   ├── main.py                    # FastAPI RAG service
│   ├── ingest.py                  # builds Chroma vector store
│   ├── knowledge_base_docs.json   # curated KB source (15 categories)
│   ├── requirements.txt
│   └── .env.example               # ANTHROPIC_API_KEY
```

## Running locally

```bash
# Backend
cd aiml_sahara
pip install -r requirements.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY
python ingest.py                # build the vector store
uvicorn main:app --reload       # http://localhost:8000

# Frontend — any static server
cd ../frontend
python -m http.server 5500      # open http://localhost:5500/landing.html
```

Set `TRIAGE_API_URL` in `assistant.html` / `askhelp.html` to your deployed
backend URL for production.

## Deployment

- **Backend**: Render / Railway / Fly.io. Chroma persists to `./chroma_db`;
  on ephemeral storage, run `ingest.py` at container start.
- **Frontend**: any static host (Firebase Hosting fits, since Firebase is
  already in the stack).
- **CORS**: restrict `allow_origins` in `main.py` to the deployed frontend
  origin before going live.
## 🚀 Deployment

### 🌐 Live Demo
- **Frontend (Firebase Hosting):** https://sahara-ai-4e3ee.web.app
- **Backend (Render):** https://sahara-ai-f4zg.onrender.com/

## Security & responsibility considerations

- The assistant explicitly disclaims that it is not a substitute for emergency
  services, medical, or law-enforcement response — in the prompt *and* the UI.
- Escalation is fail-open: uncertainty escalates rather than suppresses.
- Firebase security rules should restrict `help_requests` reads to volunteers
  and admins (requester PII + live location is sensitive).
- API keys live in environment variables only; the frontend never touches the
  LLM directly.

## Challenges & what I learned

- **Grounding vs. helpfulness tension**: a strictly grounded model refuses too
  much; an unconstrained one invents medical advice. The resolution was a
  three-tier prompt: answer from retrieval when possible → general safety
  principles with an explicit "no KB match" note → always surface emergency
  services for critical cases.
- **False-alarm design**: severity is a judgment call with asymmetric costs.
  Missing a real emergency is worse than a false escalation, so the prompt and
  tool design bias toward escalation under uncertainty, with severity + reason
  logged for review.
- **Making RAG visible**: retrieval is invisible plumbing unless the UI shows
  it. Building the explainability drawer changed how convincing the project is
  in demos.

## Roadmap

- [ ] **Evaluation harness** — labeled test set (situation → expected
      severity), measuring escalation precision/recall. *Next priority.*
- [ ] Streaming responses (SSE from FastAPI → typewriter becomes real streaming)
- [ ] Conversation persistence to Firestore (currently in-memory per session)
- [ ] Volunteer dashboard v2 + case timeline
- [ ] Interactive map (nearby hospitals / police / volunteers)
- [ ] Hindi + code-mixed language support
- [ ] Voice input for hands-busy emergencies
- [ ] Admin analytics: escalation rates, category heatmap, KB coverage gaps
