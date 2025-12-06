from fastapi import APIRouter, HTTPException
from app.models.schemas import SemanticSearchResults, SemanticSearchRequest, SemanticSearchResponse
from app.services.weaviate_service import semantic_search

router = APIRouter(
    prefix="/search",
    tags=["Semantic Search"]
)

@router.post("/semantics", response_model=SemanticSearchResponse)
def semantic_search_endpoint(payload:SemanticSearchRequest):
    try:
        hits = semantic_search(query=payload.query, top_k=payload.top_k)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed {e}")
    
    results = [
        SemanticSearchResults(**hit) for hit in hits
    ]

    return SemanticSearchResponse(results=results)
