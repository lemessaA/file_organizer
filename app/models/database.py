from typing import Optional, List, Dict
from sqlmodel import SQLModel, Field, Relationship, JSON
from datetime import datetime
import enum


class FileCategoryEnum(str, enum.Enum):
    """File category enum for database."""
    DOCUMENTS = "Documents"
    IMAGES = "Images"
    VIDEOS = "Videos"
    AUDIO = "Audio"
    ARCHIVES = "Archives"
    CODE = "Code"
    DATA = "Data"
    EXECUTABLES = "Executables"
    OTHER = "Other"


class AgentSessionDB(SQLModel, table=True):
    """Database model for agent sessions."""
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, nullable=False, max_length=255)
    directory: str = Field(nullable=False)
    status: str = Field(default="pending", nullable=False, max_length=50)
    files_processed: int = Field(default=0)
    files_organized: int = Field(default=0)
    errors: int = Field(default=0)
    dry_run: bool = Field(default=False)
    use_llm: bool = Field(default=True)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = Field(default=None, sa_column=Field(default=None))
    error_message: Optional[str] = None

    # Relationships
    decisions: List["MemoryDecisionDB"] = Relationship(back_populates="session")


class MemoryDecisionDB(SQLModel, table=True):
    """ Database model for memory decisions."""
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(nullable=False, max_length=500, index=True)
    original_path: str = Field(nullable=False)
    target_path: str = Field(nullable=False)
    category: FileCategoryEnum = Field(nullable=False, sa_column_kwargs={"type_": "ENUM"})
    decision_source: str = Field(nullable=False, max_length=50)  # extension, llm, memory, user
    confidence: float = Field(default=1.0)
    session_id: Optional[str] = Field(default=None, foreign_key="agentsessiondb.session_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    # Relationships
    session: Optional[AgentSessionDB] = Relationship(back_populates="decisions")


class FilePatternRuleDB(SQLModel, table=True):
    """Database model for file pattern rules."""
    id: Optional[int] = Field(default=None, primary_key=True)
    pattern: str = Field(nullable=False, max_length=255)
    category: FileCategoryEnum = Field(nullable=False, sa_column_kwargs={"type_": "ENUM"})
    priority: int = Field(default=0)
    is_regex: bool = Field(default=False)
    created_by: Optional[str] = Field(default=None, max_length=255)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class AgentConfigurationDB(SQLModel, table=True):
    """Database model for agent configuration."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, nullable=False, max_length=255)
    categories: Dict = Field(nullable=False, sa_column=Field(default=None))
    ignore_patterns: Optional[Dict] = Field(default=None)
    max_file_size: int = Field(default=100 * 1024 * 1024)  # 100MB
    default_confidence_threshold: float = Field(default=0.7)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    is_default: bool = Field(default=False)
