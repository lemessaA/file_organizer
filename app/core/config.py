"""Configuration with pydantic Settings. """

from typing import Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import json



class Settings(BaseSettings):
    """Application settings."""
    
    #API  
    APP_NAME: str = "fILE Organizer Agent"
    APP_ENV: str = "development"
    DEBUG:bool = False
    SECRET_KEY: str = Field(default="")
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Langsmith 
    LANGSMITH_API_KEY:str = Field(default="")
    LANGSMITH_PROJECT:str = "file-organizer-agent"
    LANGSMITH_TRACING:bool = True
    OPENAI_API_KEY: str = Field(default="")
    
    
    # Database
    DATABASE_URL: str = Field(default="postgresql://posstgres:qaws@localhost:5432/file_db")
    
    # File System 
    BASE_DIR: str = "/app" # base directory for the application(inside Docker or local)
    UPLOAD_DIR:str = "/app/processed" # upload directory where uploaded file are stored 
    PROCESSED_DIR: str = "/app/processed" # Directory where upload processed files are removed
    MAX_FILE_SIZE: int = 100*1024*1024  # 100MB
    
    #Agent Settings
    DRY_RUN: bool = False # if True, agent simulate actions without modifying  files 
    BATCH_SIZE:int = 50 # Number of files processed in one batch
    CONCURRENT_WORKERS: int = 3 # Number of parallel workers for processing
    
    
    # JSON string defining file categories and extensions
    # eg. {"images": [".png", "jpg"], "Docs": [".pdf"]}
    
    FILE_CATEGORIES_JSON: str = Field(default='{}')
    
    @property
    def file_categories(self) -> Dict[str, List[str]]:
        """
         Convert FILE_CATEGORIES_JSON into a Python dictionary.
         
          If JSON is invalid or empty, fallback to default categories.
        
        """
        try:
            # parse JSON string from environment 
            return json.loads(self.FILE_CATEGORIES_JSON)
        except json.JSONDecodeError:
            # Fallback categories if JSON is malformed 
            return {
                "Documents": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".md"],
                "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
                "Videos": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
                "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
                "Archives": [".zip", ".rar", ".tar", ".gz", ".7z"],
                "Code": [".py", ".js", ".java", ".cpp", ".c", ".html", ".css", ".json"],
                "Data": [".csv", ".xlsx", ".xls", ".tsv", ".sql", ".db"],
                "Executables": [".exe", ".msi", ".app", ".sh", ".bat"],
                "Other": []
            }
            
# MONITORING & HEALTH 

# port where matrics (prometheus, etc.) are exposed
METRICS_PORT: int = 9090

# Interval (seconds) for health checks

HEALTH_CHECK_INTERVAL: int = 30 


# pydantic config

class Config:
    # Load environment variables form .env file
    
    env_file = ".env" 
    
    case_sensitive = True
    
# Import this everywhere instead of reloading env vars

settings = Settings() 
 
    