"""
File system operations service.
"""
import os
import shutil
import mimetypes
import filetype
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.core.config import settings
from app.models.schemas import FileMetadata
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FileService:
    """Service for file system operations."""
    
    def __init__(self):
        self.base_path = Path(settings.BASE_DIR)
        self.upload_path = Path(settings.UPLOAD_DIR)
        self.processed_path = Path(settings.PROCESSED_DIR)
        self.ensure_directories()
    
    def ensure_directories(self):
        """ Ensure required directories exist."""
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def scan_directory(
        self, 
        directory: Path, 
        recursive: bool = True,
        max_depth: Optional[int] = None
    ) -> List[FileMetadata]:
        """
        Scan directory and collect file metadata.
        
        Args:
            directory: Path to scan
            recursive: Whether to scan recursively
            max_depth: Maximum recursion depth
            
        Returns:
            List of file metadata
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        files_metadata = []
        directory = directory.resolve()
        
        def _scan(current_dir: Path, depth: int):
            if max_depth is not None and depth > max_depth:
                return
            
            try:
                for item in current_dir.iterdir():
                    if item.is_dir() and recursive:
                        _scan(item, depth + 1)
                    elif item.is_file():
                        try:
                            metadata = self.get_file_metadata(item)
                            files_metadata.append(metadata)
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Could not access {item}: {e}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not access directory {current_dir}: {e}")
        
        _scan(directory, 0)
        logger.info(f"Scanned {len(files_metadata)} files in {directory}")
        return files_metadata
    
    def get_file_metadata(self, file_path: Path) -> FileMetadata:
        """
        Get metadata for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileMetadata object
        """
        stat = file_path.stat()
        
        # Try to determine MIME type
        mime_type = None
        try:
            # Try filetype first (more accurate)
            kind = filetype.guess(str(file_path))
            if kind:
                mime_type = kind.mime
            else:
                # Fallback to mimetypes
                mime_type, _ = mimetypes.guess_type(str(file_path))
        except Exception:
            pass
        
        return FileMetadata(
            path=str(file_path.resolve()),
            name=file_path.name,
            size=stat.st_size,
            extension=file_path.suffix.lower(),
            mime_type=mime_type,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            created=datetime.fromtimestamp(stat.st_ctime),
            is_hidden=file_path.name.startswith('.')
        )
    
    def safe_move(
        self, 
        source: Path, 
        target: Path, 
        dry_run: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Safely move a file with conflict resolution.
        
        Args:
            source: Source file path
            target: Target file path
            dry_run: If True, don't actually move
            
        Returns:
            Tuple of (success, error_message)
        """
        source = source.resolve()
        target = target.resolve()
        
        # Validate source
        if not source.exists():
            return False, f"Source file does not exist: {source}"
        
        if not source.is_file():
            return False, f"Source is not a file: {source}"
        
        # Ensure target directory exists
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle duplicates
        if target.exists():
            # Generate unique filename
            counter = 1
            stem = target.stem
            suffix = target.suffix
            
            while target.exists():
                target = target.parent / f"{stem}_{counter}{suffix}"
                counter += 1
        
        if dry_run:
            logger.info(f"[DRY RUN] Would move: {source} -> {target}")
            return True, None
        
        try:
            # Use shutil.move for cross-platform compatibility
            shutil.move(str(source), str(target))
            logger.info(f"Moved file: {source} -> {target}")
            return True, None
        except (shutil.Error, OSError, PermissionError) as e:
            error_msg = f"Failed to move {source} to {target}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def safe_copy(
        self, 
        source: Path, 
        target: Path, 
        dry_run: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Safely copy a file."""
        if dry_run:
            logger.info(f"[DRY RUN] Would copy: {source} -> {target}")
            return True, None
        
        try:
            shutil.copy2(str(source), str(target))
            logger.info(f"Copied file: {source} -> {target}")
            return True, None
        except (shutil.Error, OSError, PermissionError) as e:
            error_msg = f"Failed to copy {source} to {target}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file hash for duplicate detection."""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def get_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory."""
        total_size = 0
        
        for item in directory.rglob('*'):
            if item.is_file():
                try:
                    total_size += item.stat().st_size
                except OSError:
                    continue
        
        return total_size


class FileWatcher(FileSystemEventHandler):
    """Watch for new files in a directory."""
    
    def __init__(self, callback):
        self.callback = callback
        self.observer = Observer()
    
    def on_created(self, event):
        if not event.is_directory:
            self.callback(Path(event.src_path))
    
    def start(self, directory: Path):
        """Start watching directory."""
        self.observer.schedule(self, str(directory), recursive=True)
        self.observer.start()
        logger.info(f"Started watching directory: {directory}")
    
    def stop(self):
        """Stop watching."""
        self.observer.stop()
        self.observer.join()
        logger.info("File watcher stopped")