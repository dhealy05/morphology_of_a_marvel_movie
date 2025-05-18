import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List

@dataclass
class Section:
    """
    A class representing a section of a narrative with associated metadata.

    Attributes:
        text (str): The actual text content of the section
        narrative_id (str): Identifier for the narrative this section belongs to
        section_index (int): Index of this section within its narrative
        metadata (Dict[str, Any]): Additional metadata for this section
    """
    text: str
    narrative_id: str
    section_index: int
    embedding: Union[np.ndarray, List[np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the section"""
        return f"[{self.narrative_id}:{self.section_index}] {self.text[:50]}..."

    def __repr__(self) -> str:
        """Detailed representation of the section"""
        return f"Section(narrative_id='{self.narrative_id}', section_index={self.section_index}, text_len={len(self.text)}, metadata={self.metadata})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary format"""
        return {
            "text": self.text,
            "narrative_id": self.narrative_id,
            "section_index": self.section_index,
            "embedding": self.embedding,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Section':
        """Create a section from dictionary data"""
        return cls(
            text=data["text"],
            narrative_id=data["narrative_id"],
            section_index=data["section_index"],
            embedding=data.get("embedding", np.zeros((3072,))),
            metadata=data.get("metadata", {})            
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update metadata for this section"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with a default if not found"""
        return self.metadata.get(key, default)

    def get_display_label(self, max_length: int = 100) -> str:
        """
        Get a display label for this section that includes narrative ID and section number
        followed by truncated text if needed.
        """
        prefix = f"[{self.narrative_id}:{self.section_index+1}] "
        remaining = max_length - len(prefix)

        if remaining <= 0:
            return prefix

        if len(self.text) > remaining:
            truncated = self.text[:remaining-3] + "..."
        else:
            truncated = self.text

        return prefix + truncated
