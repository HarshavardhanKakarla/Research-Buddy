"""
schemas.py
----------
Pydantic models describing the structured shapes the LLM must return.
Using LangChain's structured-output binding means we get validated,
predictable JSON back instead of parsing free-form text.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ComparisonRow(BaseModel):
    aspect: str = Field(description="The dimension being compared, e.g. 'Accuracy'")
    values: List[str] = Field(
        description="One value per item being compared, in the same order as 'items'"
    )


class ComparisonTable(BaseModel):
    items: List[str] = Field(description="Names of the things being compared, e.g. paper titles or methods")
    rows: List[ComparisonRow]


class ProsCons(BaseModel):
    subject: str = Field(description="What the pros/cons are about")
    pros: List[str]
    cons: List[str]


class BulletSummary(BaseModel):
    title: str
    bullets: List[str] = Field(description="Concise, information-dense bullet points")


class StructuredAnswer(BaseModel):
    """Top-level container returned for every query, so the UI can render
    whichever fields are populated without guessing the response shape."""

    summary: str = Field(description="A short 2-4 sentence plain-language summary")
    bullets: Optional[BulletSummary] = None
    comparison: Optional[ComparisonTable] = None
    pros_cons: Optional[ProsCons] = None
    sources: List[str] = Field(default_factory=list, description="Source file names cited")
