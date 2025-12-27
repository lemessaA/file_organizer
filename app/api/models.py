"""
Pydantic models for API requests and responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class AgentStatus(str, Enum):
    """Agent status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OrganizeRequest(BaseModel):
    """Request to organize a directory."""
    directory: str = Field(..., description="Directory path to organize")
    recursive: bool = Field(True, description="Scan recursively")
    dry_run: bool = Field(False, description="Dry run mode")
    use_llm: bool = Field(True, description="Use LLM for classification")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Confidence threshold")
    batch_size: Optional[int] = Field(None, gt=0, le=1000, description="Batch size")
    categories_filter: Optional[List[str]] = Field(None, description="Categories to include")
    
    @validator('directory')
    def validate_directory(cls, v):
        import os
        if not os.path.exists(v):
            raise ValueError(f"Directory does not exist: {v}")
        if not os.path.isdir(v):
            raise ValueError(f"Path is not a directory: {v}")
        return v


class OrganizeResponse(BaseModel):
    """Response for organize request."""
    session_id: str
    status: AgentStatus
    message: str
    estimated_time: Optional[int] = Field(None, description="Estimated time in seconds")
    result_url: Optional[str] = Field(None, description="URL to get results")


class SessionStatus(BaseModel):
    """Session status."""
    session_id: str
    status: AgentStatus
    directory: str
    started_at: datetime
    completed_at: Optional[datetime]
    files_processed: int = 0
    files_organized: int = 0
    errors: int = 0
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_phase: Optional[str] = None


class SessionResult(BaseModel):
    """Session result."""
    session_id: str
    directory: str
    started_at: datetime
    completed_at: datetime
    files_processed: int
    files_organized: int
    errors: int
    categories_used: Dict[str, int]
    decision_sources: Dict[str, int]
    dry_run: bool
    classification_history: List[Dict[str, Any]]
    action_history: List[Dict[str, Any]]
    stored_decisions: List[Dict[str, Any]]


class MemoryStats(BaseModel):
    """Memory statistics."""
    total_decisions: int
    recent_decisions: int
    category_distribution: Dict[str, int]
    decision_sources: Dict[str, int]
    average_confidence: float
    cache_hits: int
    cache_misses: int


class UpdateDecisionRequest(BaseModel):
    """Request to update a memory decision."""
    filename: str
    new_category: str
    new_target_path: Optional[str] = None


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime: float
    database: bool
    redis: bool
    openai: bool
    langsmith: bool
    timestamp: datetime


class MetricsResponse(BaseModel):
    """Metrics response."""
    sessions_total: int
    sessions_active: int
    files_processed_total: int
    files_organized_total: int
    error_rate: float