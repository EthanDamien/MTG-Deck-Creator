import json as _json
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator


# ─── Parsed query ────────────────────────────────────────────
class ParsedQuery(BaseModel):
    format: Literal["commander", "60card"] = "commander"
    commander: Optional[str] = None
    colors: Optional[list[Literal["W", "U", "B", "R", "G"]]] = None
    theme: str = "general"
    style: Literal["aggro", "midrange", "control", "combo", "tribal"] = "midrange"
    constraints: list[str] = []
    pinned_cards: list[str] = []
    banned_cards: list[str] = []

    @field_validator("format", mode="before")
    @classmethod
    def coerce_format(cls, v: Any) -> Any:
        if not v:
            return "commander"
        return v

    @field_validator("style", mode="before")
    @classmethod
    def coerce_style(cls, v: Any) -> Any:
        if not v:
            return "midrange"
        valid = {"aggro", "midrange", "control", "combo", "tribal"}
        if isinstance(v, str) and v.lower() in valid:
            return v.lower()
        return "midrange"

    @field_validator("colors", "constraints", "pinned_cards", "banned_cards", mode="before")
    @classmethod
    def coerce_string_to_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            try:
                parsed = _json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return []
        return v


_ROLE_TO_WAVE = {
    "commander": 1, "theme": 1,
    "ramp": 2, "draw": 2, "protection": 2,
    "removal": 3, "wipes": 3,
    "lands": 4,
}

# ─── Slot manifest ───────────────────────────────────────────
class Slot(BaseModel):
    role: str
    count: int
    hint: str
    wave: int = 1

    @field_validator("wave", mode="before")
    @classmethod
    def default_wave_from_role(cls, v: Any, info: Any) -> int:
        if v is not None and v != 0:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        role = (info.data or {}).get("role", "")
        return _ROLE_TO_WAVE.get(role, 2)


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
_VALID_CATEGORIES = {"mana", "curve", "strategy", "synergy", "redundancy", "legality"}
_VALID_SEVERITIES = {"critical", "high", "medium"}
_VALID_FIX_ACTIONS = {"REPLACE", "ADD", "REMOVE"}


class Issue(BaseModel):
    severity: Literal["critical", "high", "medium"]
    category: Literal["mana", "curve", "strategy", "synergy", "redundancy", "legality"]
    issue: str
    detail: str
    fix_action: Literal["REPLACE", "ADD", "REMOVE"]
    target_card: Optional[str] = None
    slot: Optional[str] = None
    fix_hint: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v: Any) -> Any:
        if isinstance(v, str) and v.lower() in _VALID_SEVERITIES:
            return v.lower()
        return "medium"

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_low = v.lower()
            if v_low in _VALID_CATEGORIES:
                return v_low
            if "mana" in v_low or "ramp" in v_low:
                return "mana"
            if "curve" in v_low:
                return "curve"
            if "synergy" in v_low or "combo" in v_low:
                return "synergy"
            if "redund" in v_low:
                return "redundancy"
            if "legal" in v_low or "ban" in v_low:
                return "legality"
        return "strategy"

    @field_validator("fix_action", mode="before")
    @classmethod
    def coerce_fix_action(cls, v: Any) -> Any:
        if isinstance(v, str) and v.upper() in _VALID_FIX_ACTIONS:
            return v.upper()
        return "REPLACE"


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
