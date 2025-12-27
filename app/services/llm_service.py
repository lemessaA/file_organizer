"""
LLM service for file classification.
"""
from typing import Dict, List, Optional, Tuple
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage, SystemMessage

from app.core.config import settings
from app.models.schemas import ClassificationResult, FileCategory
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """Service for LLM-based file classification."""
    
    def __init__(self):
        self.llm = None
        self.parser = PydanticOutputParser(pydantic_object=ClassificationResult)
        
        if settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1,
                max_tokens=200,
            )
            logger.info(f"LLM service initialized with model: {settings.OPENAI_MODEL}")
        else:
            logger.warning("OpenAI API key not found. LLM classification disabled.")
    
    def classify_by_filename(
        self, 
        filename: str, 
        extension: str,
        categories: List[str]
    ) -> Optional[ClassificationResult]:
        """
        Classify file using LLM based on filename.
        
        Args:
            filename: Name of the file
            extension: File extension
            categories: Available categories
            
        Returns:
            ClassificationResult or None
        """
        if not self.llm:
            return None
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a file classification assistant. 
            Analyze the filename and suggest the most appropriate category.
            Return valid JSON with category, confidence, and reasoning."""),
            ("human", """
            Filename: {filename}
            Extension: {extension}
            
            Available categories: {categories}
            
            Guidelines:
            1. Consider the filename semantics (e.g., "report_final.pdf" → Documents)
            2. Consider common patterns (e.g., "IMG_" prefix → Images)
            3. Return confidence between 0.0 and 1.0
            4. Provide brief reasoning
            
            Example response:
            {{
                "category": "Documents",
                "confidence": 0.85,
                "reasoning": "Filename contains 'report' which suggests it's a document",
                "source": "llm"
            }}
            """),
        ])
        
        try:
            # Format the prompt
            formatted_prompt = prompt.format_messages(
                filename=filename,
                extension=extension,
                categories=json.dumps(categories)
            )
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            # Parse the response
            result = self.parser.parse(response.content)
            
            # Validate category
            try:
                result.category = FileCategory(result.category)
            except ValueError:
                # If category not in enum, use Other
                result.category = FileCategory.OTHER
            
            logger.info(f"LLM classified '{filename}' as '{result.category}' "
                       f"(confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"LLM classification failed for '{filename}': {e}")
            return None
    
    def batch_classify(
        self, 
        files: List[Dict], 
        batch_size: int = 10
    ) -> List[Optional[ClassificationResult]]:
        """
        Classify multiple files in batch.
        
        Args:
            files: List of dicts with filename and extension
            batch_size: Number of files per batch
            
        Returns:
            List of classification results
        """
        if not self.llm:
            return [None] * len(files)
        
        results = []
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            try:
                # Create batch prompt
                batch_prompt = self._create_batch_prompt(batch)
                response = self.llm.invoke(batch_prompt)
                
                # Parse batch response
                batch_results = self._parse_batch_response(response.content, len(batch))
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Batch classification failed: {e}")
                # Add None for each failed classification
                results.extend([None] * len(batch))
        
        return results
    
    def _create_batch_prompt(self, batch: List[Dict]) -> List:
        """Create prompt for batch classification."""
        files_text = "\n".join([
            f"{i+1}. {file['filename']} (extension: {file['extension']})"
            for i, file in enumerate(batch)
        ])
        
        return [
            SystemMessage(content="""You are a batch file classification assistant.
            Classify each file and return a JSON array of results.
            Each result should have category, confidence, and reasoning."""),
            HumanMessage(content=f"""Classify these files:
            {files_text}
            
            Available categories: {list(FileCategory)}
            
            Return format:
            [
                {{
                    "category": "Documents",
                    "confidence": 0.85,
                    "reasoning": "Brief explanation",
                    "source": "llm"
                }},
                ...
            ]
            """)
        ]
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[Optional[ClassificationResult]]:
        """Parse batch response from LLM."""
        try:
            data = json.loads(response)
            results = []
            
            for item in data[:expected_count]:
                try:
                    result = ClassificationResult(**item)
                    results.append(result)
                except Exception:
                    results.append(None)
            
            # Pad if response has fewer items
            while len(results) < expected_count:
                results.append(None)
            
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse batch response: {e}")
            return [None] * expected_count