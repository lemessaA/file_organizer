"""
Memory service for storing and retrieving decisions.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from app.database.session import get_redis
from app.models.database import MemoryDecisionDB, AgentSessionDB, FileCategoryEnum
from app.models.schemas import MemoryDecision, FileCategory
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryService:
    """Service for decision memory with Redis cache and database persistence."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.redis = get_redis()
        self.cache_ttl = timedelta(hours=24)
    
    def remember(
        self,
        filename: str,
        original_path: str,
        target_path: str,
        category: FileCategory,
        decision_source: str,
        confidence: float,
        session_id: str
    ) -> MemoryDecision:
        """
        Store a decision in memory.
        
        Args:
            filename: Name of the file
            original_path: Original file path
            target_path: Target file path
            category: File category
            decision_source: Source of decision (extension, llm, memory, user)
            confidence: Confidence score
            session_id: Agent session ID
            
        Returns:
            MemoryDecision object
        """
        # Create database record
        db_decision = MemoryDecisionDB(
            filename=filename,
            original_path=original_path,
            target_path=target_path,
            category=FileCategoryEnum(category.value),
            decision_source=decision_source,
            confidence=confidence,
            session_id=session_id
        )
        
        self.db.add(db_decision)
        self.db.commit()
        self.db.refresh(db_decision)
        
        # Cache in Redis
        cache_key = f"decision:{filename}"
        decision_data = {
            "filename": filename,
            "original_path": original_path,
            "target_path": target_path,
            "category": category.value,
            "decision_source": decision_source,
            "confidence": confidence,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis.setex(
            cache_key,
            int(self.cache_ttl.total_seconds()),
            json.dumps(decision_data)
        )
        
        # Convert to schema
        decision = MemoryDecision(
            filename=filename,
            original_path=original_path,
            target_path=target_path,
            category=category,
            decision_source=decision_source,
            confidence=confidence,
            timestamp=datetime.now(),
            session_id=session_id
        )
        
        logger.debug(f"Remembered decision for {filename}")
        return decision
    
    def recall(self, filename: str) -> Optional[MemoryDecision]:
        """
        Recall a decision from memory.
        
        Args:
            filename: Name of the file
            
        Returns:
            MemoryDecision or None
        """
        # Try cache first
        cache_key = f"decision:{filename}"
        cached = self.redis.get(cache_key)
        
        if cached:
            try:
                data = json.loads(cached)
                return MemoryDecision(**data)
            except (json.JSONDecodeError, ValueError):
                # Cache corrupted, fall back to database
                pass
        
        # Query database
        db_decision = self.db.query(MemoryDecisionDB).filter(
            MemoryDecisionDB.filename == filename
        ).order_by(desc(MemoryDecisionDB.created_at)).first()
        
        if db_decision:
            # Update cache
            decision_data = {
                "filename": db_decision.filename,
                "original_path": db_decision.original_path,
                "target_path": db_decision.target_path,
                "category": db_decision.category.value,
                "decision_source": db_decision.decision_source,
                "confidence": db_decision.confidence,
                "session_id": db_decision.session_id,
                "timestamp": db_decision.created_at.isoformat()
            }
            
            self.redis.setex(
                cache_key,
                int(self.cache_ttl.total_seconds()),
                json.dumps(decision_data)
            )
            
            # Convert to schema
            return MemoryDecision(
                filename=db_decision.filename,
                original_path=db_decision.original_path,
                target_path=db_decision.target_path,
                category=FileCategory(db_decision.category.value),
                decision_source=db_decision.decision_source,
                confidence=db_decision.confidence,
                timestamp=db_decision.created_at,
                session_id=db_decision.session_id
            )
        
        return None
    
    def recall_by_pattern(
        self, 
        pattern: str, 
        use_regex: bool = False
    ) -> List[MemoryDecision]:
        """
        Recall decisions matching a pattern.
        
        Args:
            pattern: Pattern to match
            use_regex: Whether pattern is a regex
            
        Returns:
            List of matching decisions
        """
        if use_regex:
            # PostgreSQL regex match
            db_decisions = self.db.query(MemoryDecisionDB).filter(
                MemoryDecisionDB.filename.op('~')(pattern)
            ).all()
        else:
            # Simple LIKE match
            db_decisions = self.db.query(MemoryDecisionDB).filter(
                MemoryDecisionDB.filename.like(f"%{pattern}%")
            ).all()
        
        return [
            MemoryDecision(
                filename=d.filename,
                original_path=d.original_path,
                target_path=d.target_path,
                category=FileCategory(d.category.value),
                decision_source=d.decision_source,
                confidence=d.confidence,
                timestamp=d.created_at,
                session_id=d.session_id
            )
            for d in db_decisions
        ]
    
    def forget(self, filename: str) -> bool:
        """
        Remove a decision from memory.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if successful
        """
        # Remove from database
        deleted = self.db.query(MemoryDecisionDB).filter(
            MemoryDecisionDB.filename == filename
        ).delete()
        
        if deleted:
            self.db.commit()
            
            # Remove from cache
            cache_key = f"decision:{filename}"
            self.redis.delete(cache_key)
            
            logger.info(f"Forgot decision for {filename}")
            return True
        
        return False
    
    def update_decision(
        self,
        filename: str,
        new_category: FileCategory,
        new_target: Optional[str] = None
    ) -> Optional[MemoryDecision]:
        """
        Update an existing decision.
        
        Args:
            filename: Name of the file
            new_category: New category
            new_target: New target path (optional)
            
        Returns:
            Updated MemoryDecision or None
        """
        db_decision = self.db.query(MemoryDecisionDB).filter(
            MemoryDecisionDB.filename == filename
        ).first()
        
        if not db_decision:
            return None
        
        # Update fields
        db_decision.category = FileCategoryEnum(new_category.value)
        db_decision.decision_source = "user"
        db_decision.confidence = 1.0
        
        if new_target:
            db_decision.target_path = new_target
        
        self.db.commit()
        
        # Update cache
        cache_key = f"decision:{filename}"
        decision_data = {
            "filename": db_decision.filename,
            "original_path": db_decision.original_path,
            "target_path": db_decision.target_path,
            "category": db_decision.category.value,
            "decision_source": db_decision.decision_source,
            "confidence": db_decision.confidence,
            "session_id": db_decision.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis.setex(
            cache_key,
            int(self.cache_ttl.total_seconds()),
            json.dumps(decision_data)
        )
        
        return MemoryDecision(
            filename=db_decision.filename,
            original_path=db_decision.original_path,
            target_path=db_decision.target_path,
            category=FileCategory(db_decision.category.value),
            decision_source=db_decision.decision_source,
            confidence=db_decision.confidence,
            timestamp=datetime.now(),
            session_id=db_decision.session_id
        )
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Statistics dictionary
        """
        from datetime import date
        cutoff_date = date.today() - timedelta(days=days)
        
        # Total decisions
        total = self.db.query(MemoryDecisionDB).count()
        
        # Recent decisions
        recent = self.db.query(MemoryDecisionDB).filter(
            MemoryDecisionDB.created_at >= cutoff_date
        ).count()
        
        # Category distribution
        category_counts = {}
        for category in FileCategoryEnum:
            count = self.db.query(MemoryDecisionDB).filter(
                MemoryDecisionDB.category == category
            ).count()
            category_counts[category.value] = count
        
        # Decision sources
        source_counts = {}
        sources = self.db.query(
            MemoryDecisionDB.decision_source,
            func.count(MemoryDecisionDB.id)
        ).group_by(MemoryDecisionDB.decision_source).all()
        
        for source, count in sources:
            source_counts[source] = count
        
        # Average confidence
        avg_confidence = self.db.query(
            func.avg(MemoryDecisionDB.confidence)
        ).scalar() or 0.0
        
        return {
            "total_decisions": total,
            "recent_decisions": recent,
            "category_distribution": category_counts,
            "decision_sources": source_counts,
            "average_confidence": float(avg_confidence),
            "cache_hits": self.redis.info()['keyspace_hits'],
            "cache_misses": self.redis.info()['keyspace_misses']
        }