from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role:         str
    content:      str | list | None = None
    tool_calls:   list[dict] | None = None
    tool_call_id: str | None = None
    name:         str | None = None
    model_config  = ConfigDict(extra="allow")


class OpenAIChatRequest(BaseModel):
    model:       str = "optmod-router"
    messages:    list[ChatMessage]
    tools:       list[dict] | None = None
    tool_choice: str | dict | None = None
    stream:      bool = False
    temperature: float | None = None
    max_tokens:  int | None = None
    model_config = ConfigDict(extra="allow")


@dataclass
class Features:
    task_type:         str
    difficulty:        str
    token_count:       int
    has_tools:         bool
    language:          str
    last_user_message: str


@dataclass
class SessionPin:
    model_name:         str
    last_turn_at:       float
    last_cache_rate:    float = 0.0
    last_prompt_tokens: int   = 0
    turn_count:         int   = 0


@dataclass
class RoutingContext:
    request:         OpenAIChatRequest
    features:        Features
    session_id:      str
    registry:        "ModelRegistry"
    attempt_number:  int = 0
    last_error_type: str | None = None
    models_tried:    list[str] = field(default_factory=list)
    session_pin:     SessionPin | None = None


@dataclass
class RoutingDecision:
    model:       "ModelConfig"
    mutator:     str
    reason:      str
    confidence:  float
    router_name: str
    meta:        dict = field(default_factory=dict)


@dataclass
class LogEntry:
    ts:                str
    session_id:        str
    task_type:         str
    difficulty:        str
    token_count:       int
    has_tools:         bool
    language:          str
    router:            str
    decision_model:    str
    decision_reason:   str
    confidence:        float
    mutator:           str
    models_tried:      list[str]
    escalation_count:  int
    final_model:       str
    ok:                bool
    error_type:        str | None
    latency_ms:        float
    prompt_tokens:     int
    completion_tokens: int
    cached_tokens:     int = 0
    pin_state:         str = "fresh"
    # PerfRouter / Laya tier diagnostics (defaults for other routers and pins)
    pr_task_type:       str = ""
    pr_routing_mode:    str = ""
    pr_top_similarity:  float = 0.0
    laya_status:        str = ""
    laya_tier:          str = ""
    laya_confidence:    float = 0.0
    laya_ms:            float = 0.0
    laya_cached:        bool = False   # tier served from the classifier memo; laya_ms is then ~0
    laya_applied:       bool = False
    laya_shadow_model:  str = ""
    effective_delta:    float | None = None
    effective_cost_cap: float | None = None

    def to_jsonl(self) -> str:
        return json.dumps(dataclasses.asdict(self))
