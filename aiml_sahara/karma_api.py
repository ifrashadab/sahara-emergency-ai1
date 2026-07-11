# karma_api.py — server-side karma awards (never trust the client with points)
#
# Setup (one time):
#   1. pip3 install firebase-admin --user
#   2. Firebase console → Project settings → Service accounts →
#      "Generate new private key" → save the JSON as
#      aiml_sahara/serviceAccountKey.json
#      (ADD serviceAccountKey.json TO .gitignore — it is a secret!)
#   3. In main.py add two lines:
#        from karma_api import router as karma_router
#        app.include_router(karma_router)
#
# Security model: the client sends its Firebase ID token; we verify it,
# confirm the caller is actually the volunteer assigned to the case, check
# the case status matches the action, and only then award points with the
# Admin SDK. Duplicate awards are blocked with an idempotency marker.

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, auth as fb_auth, firestore

router = APIRouter()

_KEY_PATH = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(_KEY_PATH))
db = firestore.client()

POINTS = {
    "accept": 100,
    "resolve": 250,
    "resolve_critical_bonus": 150,
    "profile_complete": 50,
}


class KarmaRequest(BaseModel):
    idToken: str
    action: str   # "accept" | "resolve"
    caseId: str


@router.post("/karma/award")
def award_karma(req: KarmaRequest):
    # 1. Verify the caller's identity
    try:
        decoded = fb_auth.verify_id_token(req.idToken)
    except Exception:
        raise HTTPException(401, "Invalid auth token")
    uid = decoded["uid"]

    if req.action not in ("accept", "resolve"):
        raise HTTPException(400, "Unknown action")

    case_ref = db.collection("help_requests").document(req.caseId)
    case = case_ref.get()
    if not case.exists:
        raise HTTPException(404, "Case not found")
    c = case.to_dict()

    # 2. The caller must be the assigned volunteer
    if c.get("assignedTo") != uid:
        raise HTTPException(403, "Not your case")

    # 3. Case status must match the action
    expected = {"accept": "assigned", "resolve": "resolved"}[req.action]
    if c.get("status") != expected:
        raise HTTPException(409, f"Case is not in '{expected}' state")

    # 4. Idempotency — each action awards once per case
    marker = f"karma_{req.action}_awarded"
    if c.get(marker):
        return {"awarded": 0, "reason": "already awarded"}

    points = POINTS[req.action]
    if req.action == "resolve" and c.get("severity") == "critical":
        points += POINTS["resolve_critical_bonus"]

    # 5. Atomic award
    user_ref = db.collection("users").document(uid)

    @firestore.transactional
    def txn(transaction):
        snap = user_ref.get(transaction=transaction)
        k = (snap.to_dict() or {}).get("karma", {})
        transaction.set(user_ref, {
            "karma": {
                "lifetime": (k.get("lifetime", 0)) + points,
                "spendable": (k.get("spendable", 0)) + points,
            }
        }, merge=True)
        transaction.update(case_ref, {marker: True})

    txn(db.transaction())

    return {"awarded": points, "action": req.action}
