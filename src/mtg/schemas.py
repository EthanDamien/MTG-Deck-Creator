import json as _json
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator


# ─── Parsed query ────────────────────────────────────────────
class ParsedQuery(BaseModel):
    format: Literal["commander", "60card"]
    commander: Optional[str] = None
    colors: Optional[list[Literal["W", "U", "B", "R", "G"]]] = None
    theme: str
    style: Literal["aggro", "midrange", "control", "combo", "tribal"]
    constraints: list[str] = []
    pinned_cards: list[str] = []
    banned_cards: list[str] = []

    @field_validator("colors", "constraints", "pinned_cards", "banned_cards", mode="before")
    @classmethod
    def coerce_string_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                parsed = _json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return v


# ─── Slot manifest ───────────────────────────────────────────
class Slot(BaseModel):
    role: str
    count: int
    hint: str
    wave: int  # 1, 2, 3, 4


class DeckPlan(BaseModel):
    template: str
    slots: list[Slot]


# ─── Worker output ───────────────────────────────────────────
class SubQuery(BaseModel):
    role: str
    query: str  # dense-phrase RAG query


class WorkerOutput(BaseModel):
    role: str
    queries: list[SubQuery]


# ─── Card record (in DB) ─────────────────────────────────────
class CardRecord(BaseModel):
    name: str
    oracle_id: str
    mana_cost: str
    cmc: float
    colors: list[str]
    color_identity: list[str]
    type_line: str
    oracle_text: str
    keywords: list[str]
    power: Optional[str] = None
    toughness: Optional[str] = None
    edhrec_rank: Optional[int] = None
    legalities: dict[str, str]
    image_uri: Optional[str] = None
    reasoning: str  # the dense-phrase RAG document


# ─── Picker ──────────────────────────────────────────────────
class Pick(BaseModel):
    slot: str
    card: str
    reason: str


class PickerOutput(BaseModel):
    picks: list[Pick]


# ─── Validation ──────────────────────────────────────────────
class Issue(BaseModel):
    severity: Literal["critical", "high", "medium"]
    category: Literal["mana", "curve", "strategy", "synergy", "redundancy", "legality"]
    issue: str
    detail: str
    fix_action: Literal["REPLACE", "ADD", "REMOVE"]
    target_card: Optional[str] = None
    slot: Optional[str] = None
    fix_hint: str


class ValidationReport(BaseModel):
    passed: bool
    issues: list[Issue]


# ─── Server request/response ─────────────────────────────────
class BuildRequest(BaseModel):
    query: str


class BuildResponse(BaseModel):
    deck: list[Pick]
    plan: Optional[DeckPlan] = None
    issues: list[Issue] = []
