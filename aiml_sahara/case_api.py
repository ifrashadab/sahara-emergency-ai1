# case_api.py — requester-side case flow + server-side karma & ratings.
#
# Replaces the old karma_api flow. Wire into main.py with:
#     from case_api import router as case_router
#     app.include_router(case_router)
# (Remove the old karma_api include if present.)
#
# Requires serviceAccountKey.json next to this file (already set up) and
# firebase-admin installed.
#
# Flow:
#   open --(volunteers write offers client-side)--> requester chooses
#   POST /case/choose  -> status: assigned, phone shared, +100 to volunteer
#   volunteer taps "I've helped" (client-side) -> status: pending_confirmation
#   POST /case/confirm -> status: resolved, rating stored, +250 (+150 crit)
#   POST /case/auto_confirm (volunteer, 24h elapsed) -> same as confirm, no rating

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, auth as fb_auth, firestore

router = APIRouter()

_KEY_PATH = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(_KEY_PATH))
db = firestore.client()

PTS_CHOSEN = 100
PTS_RESOLVE = 250
PTS_CRIT_BONUS = 150


def _get_case(case_id: str, key: str):
    ref = db.collection("help_requests").document(case_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(404, "Case not found")
    data = snap.to_dict()
    if not key or data.get("accessKey") != key:
        raise HTTPException(403, "Invalid case key")
    return ref, data


def _award(uid: str, points: int, rating: Optional[int] = None):
    ref = db.collection("users").document(uid)
    snap = ref.get()
    d = snap.to_dict() or {}
    k = d.get("karma", {})
    update = {
        "karma": {
            "lifetime": k.get("lifetime", 0) + points,
            "spendable": k.get("spendable", 0) + points,
        },
        "resolvedCount": d.get("resolvedCount", 0) + (1 if rating is not None or points >= PTS_RESOLVE else 0),
    }
    if rating is not None:
        update["ratingSum"] = d.get("ratingSum", 0) + rating
        update["ratingCount"] = d.get("ratingCount", 0) + 1
    ref.set(update, merge=True)


# ---------- Requester endpoints (auth = secret case key) ----------

@router.get("/case/{case_id}")
def get_case(case_id: str, key: str):
    _, d = _get_case(case_id, key)
    # Expose a requester-safe view
    out = {
        "status": d.get("status"),
        "type": d.get("type"),
        "description": d.get("description"),
        "severity": d.get("severity"),
        "createdAt": str(d.get("createdAt", "")),
        "offers": [
            {
                "uid": uid,
                "name": o.get("name", "Volunteer"),
                "ratingAvg": o.get("ratingAvg"),
                "resolvedCount": o.get("resolvedCount", 0),
            }
            for uid, o in (d.get("offers") or {}).items()
        ],
        "assignedToName": d.get("assignedToName"),
        "assignedPhone": d.get("assignedPhone") if d.get("status") in ("assigned", "pending_confirmation") else None,
    }
    return out


class ChooseReq(BaseModel):
    caseId: str
    key: str
    volunteerUid: str


@router.post("/case/choose")
def choose_volunteer(req: ChooseReq):
    ref, d = _get_case(req.caseId, req.key)
    if d.get("status") != "open":
        raise HTTPException(409, "Case is not open")
    offers = d.get("offers") or {}
    if req.volunteerUid not in offers:
        raise HTTPException(400, "That volunteer has not offered")

    vsnap = db.collection("users").document(req.volunteerUid).get()
    v = vsnap.to_dict() or {}
    ref.update({
        "status": "assigned",
        "assignedTo": req.volunteerUid,
        "assignedToName": offers[req.volunteerUid].get("name") or v.get("displayName", "Volunteer"),
        "assignedPhone": v.get("phone") or None,   # shared only now
        "assignedAt": firestore.SERVER_TIMESTAMP,
    })
    _award(req.volunteerUid, PTS_CHOSEN)
    return {"ok": True, "assignedPhone": v.get("phone")}


class ConfirmReq(BaseModel):
    caseId: str
    key: str
    rating: Optional[int] = None   # 1–5


@router.post("/case/confirm")
def confirm_resolution(req: ConfirmReq):
    ref, d = _get_case(req.caseId, req.key)
    if d.get("status") not in ("assigned", "pending_confirmation"):
        raise HTTPException(409, "Case is not awaiting confirmation")
    if d.get("karma_resolve_awarded"):
        return {"ok": True, "note": "already confirmed"}

    rating = req.rating if req.rating and 1 <= req.rating <= 5 else None
    points = PTS_RESOLVE + (PTS_CRIT_BONUS if d.get("severity") == "critical" else 0)

    ref.update({
        "status": "resolved",
        "resolvedAt": firestore.SERVER_TIMESTAMP,
        "requesterConfirmed": True,
        "rating": rating,
        "karma_resolve_awarded": True,
    })
    if d.get("assignedTo"):
        _award(d["assignedTo"], points, rating=rating)
    return {"ok": True, "awarded": points}


# ---------- Volunteer endpoint (auth = Firebase ID token) ----------

class AutoConfirmReq(BaseModel):
    idToken: str
    caseId: str


@router.post("/case/auto_confirm")
def auto_confirm(req: AutoConfirmReq):
    try:
        uid = fb_auth.verify_id_token(req.idToken)["uid"]
    except Exception:
        raise HTTPException(401, "Invalid token")

    ref = db.collection("help_requests").document(req.caseId)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(404, "Case not found")
    d = snap.to_dict()

    if d.get("assignedTo") != uid:
        raise HTTPException(403, "Not your case")
    if d.get("status") != "pending_confirmation":
        raise HTTPException(409, "Not pending confirmation")
    if d.get("karma_resolve_awarded"):
        return {"ok": True, "note": "already awarded"}

    ts = d.get("confirmRequestedAt")
    if not ts or datetime.now(timezone.utc) - ts < timedelta(hours=24):
        raise HTTPException(409, "24h has not elapsed")

    points = PTS_RESOLVE + (PTS_CRIT_BONUS if d.get("severity") == "critical" else 0)
    ref.update({
        "status": "resolved",
        "resolvedAt": firestore.SERVER_TIMESTAMP,
        "requesterConfirmed": False,
        "karma_resolve_awarded": True,
    })
    _award(uid, points)
    return {"ok": True, "awarded": points, "auto": True}
