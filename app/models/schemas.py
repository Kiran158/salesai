from pydantic import BaseModel, Field
from typing import List, Optional


class SemanticSearchRequest(BaseModel):
    query:str=Field(..., description="natural language text to search for similar deals")
    top_k:int=Field(5, description="no of similar results to return")

class SemanticSearchResults(BaseModel):
    order_id: Optional[str]
    product_name: Optional[str]
    revenue_inr: Optional[float]
    deal_notes: Optional[str]
    region: Optional[str]
    customer_segment: Optional[str]
    distance: Optional[float]


class SemanticSearchResponse(BaseModel):
    results:List[SemanticSearchResults]