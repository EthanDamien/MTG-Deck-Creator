from typing_extensions import TypedDict, Annotated
from typing import Optional
from operator import add
from mtg.schemas import ParsedQuery, DeckPlan, Pick, Issue, WorkerOutput


class DeckBuildState(TypedDict):
    user_query: str
    parsed: Optional[ParsedQuery]
    plan: Optional[DeckPlan]
    current_wave: int
    worker_outputs: Annotated[list[WorkerOutput], add]  # parallel workers append; picker resets via _replace_list
    picks: Annotated[list[Pick], add]  # accumulates across waves
    used_cards: list[str]              # dedup set (as list for JSON)
    issues: list[Issue]
    repair_attempts: int
    final_deck: Optional[list[Pick]]
