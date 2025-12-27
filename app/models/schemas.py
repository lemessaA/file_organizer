"""
Pydantic schemas for API requests/responses.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class FileCategory(str, Enum):
    """File categories."""
    DOCUMENTS = "Documents"
    IMAGES = "Images"
    VIDEOS = "Videos"
    AUDIO = "Audio"
    ARCHIVES = "Archives"
    CODE = "Code"
    DATA = "Data"
    EXECUTABLES = "Executables"
    OTHER = "Other"


class FileMetadata(BaseModel):
    """File metadata schema."""
    path: str
    name: str
    size: int
    extension: str
    mime_type: Optional[str] = None
    last_modified: datetime
    created: datetime
    is_hidden: bool = False
    
    @validator('size')
    def validate_size(cls, v):
        if v < 0:
            raise ValueError("File size cannot be negative")
        return v


class ClassificationRequest(BaseModel):
    """Request for file classification."""
    file_metadata: FileMetadata
    use_llm: bool = True
    confidence_threshold: float = 0.7


class ClassificationResult(BaseModel):
    """Classification result."""
    category: FileCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    source: str = Field(default="extension")  # extension, llm, memory
    suggested_path: Optional[str] = None


class AgentAction(BaseModel):
    """Agent action schema."""
    action_type: str = Field(default="move")  # move, copy, delete, skip
    source_path: str
    target_path: Optional[str] = None
    category: Optional[FileCategory] = None
    dry_run: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentSession(BaseModel):
    """Agent session schema."""
    session_id: str
    directory: str
    status: str = Field(default="pending")  # pending, running, completed, failed
    files_processed: int = 0
    files_organized: int = 0
    errors: int = 0
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


class BatchProcessRequest(BaseModel):
    """Batch processing request."""
    directory: str
    recursive: bool = True
    dry_run: bool = False
    use_llm: bool = True
    batch_size: Optional[int] = None
    categories_filter: Optional[List[FileCategory]] = None


class BatchProcessResponse(BaseModel):
    """Batch processing response."""
    session_id: str
    status: str
    total_files: int
    processed_files: int
    estimated_time_remaining: Optional[int] = None  # seconds
    result_url: Optional[str] = None


class MemoryDecision(BaseModel):
    """Memory decision schema."""
    filename: str
    original_path: str
    target_path: str
    category: FileCategory
    decision_source: str
    confidence: float
    timestamp: datetime
    session_id: str


class AgentStatistics(BaseModel):
    """Agent statistics."""
    total_sessions: int
    total_files_processed: int
    success_rate: float
    avg_processing_time: float
    category_distribution: Dict[str, int]
    common_errors: List[str]