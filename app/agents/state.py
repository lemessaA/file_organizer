"""
State schemas for LangGraph agent.
"""
from typing import Dict, List, Optional, Any, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from enum import Enum
import operator

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

from app.models.schemas import FileMetadata, FileCategory, AgentAction


class AgentPhase(str, Enum):
    """Agent execution phases."""
    OBSERVE = "observe"
    REASON = "reason"
    DECIDE = "decide"
    ACT = "act"
    REMEMBER = "remember"
    COMPLETE = "complete"
    ERROR = "error"


class ObservationState(BaseModel):
    """State for observation phase."""
    directory: str
    recursive: bool = True
    max_depth: Optional[int] = None
    files_found: List[FileMetadata] = Field(default_factory=list)
    files_skipped: List[FileMetadata] = Field(default_factory=list)
    scan_errors: List[Dict[str, Any]] = Field(default_factory=list)
    total_scanned: int = 0


class ReasoningState(BaseModel):
    """State for reasoning phase."""
    current_file: Optional[FileMetadata] = None
    category: Optional[FileCategory] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None
    decision_source: str = "unknown"
    use_llm: bool = True
    confidence_threshold: float = 0.7
    classification_history: List[Dict[str, Any]] = Field(default_factory=list)


class DecisionState(BaseModel):
    """State for decision phase."""
    action: Optional[AgentAction] = None
    should_skip: bool = False
    skip_reason: Optional[str] = None
    duplicate_detected: bool = False
    conflict_resolution: Optional[str] = None
    decisions_made: List[AgentAction] = Field(default_factory=list)


class ActionState(BaseModel):
    """State for action phase."""
    action_executed: bool = False
    success: bool = False
    error_message: Optional[str] = None
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    dry_run: bool = False
    action_history: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryState(BaseModel):
    """State for memory phase."""
    decision_stored: bool = False
    memory_key: Optional[str] = None
    recall_used: bool = False
    memory_updated: bool = False
    stored_decisions: List[Dict[str, Any]] = Field(default_factory=list)


class AgentState(TypedDict):
    """Main agent state for LangGraph."""
    # Core state
    phase: AgentPhase
    session_id: str
    directory: str
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Component states
    observation: ObservationState
    reasoning: ReasoningState
    decision: DecisionState
    action: ActionState
    memory: MemoryState
    
    # Statistics
    files_processed: int
    files_organized: int
    errors: int
    categories_used: Annotated[Dict[str, int], operator.add]
    decision_sources: Annotated[Dict[str, int], operator.add]
    
    # Messages for LLM communication
    messages: Annotated[List[Any], add_messages]
    
    # Configuration
    config: Dict[str, Any]
    
    # Results
    result: Optional[Dict[str, Any]]
    error: Optional[str]