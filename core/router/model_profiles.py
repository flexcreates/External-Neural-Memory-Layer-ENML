from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    max_evidence_items: int
    memory_token_budget: int
    allow_distillation: bool
    strict_grounding: bool


SMALL_MODEL_PROFILE = ModelProfile(
    name="small",
    max_evidence_items=5,
    memory_token_budget=900,
    allow_distillation=False,
    strict_grounding=True,
)

MEDIUM_MODEL_PROFILE = ModelProfile(
    name="medium",
    max_evidence_items=8,
    memory_token_budget=1500,
    allow_distillation=True,
    strict_grounding=False,
)
