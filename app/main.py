from app.config import settings
from fastapi import FastAPI
from app.routers import search_router

app = FastAPI(title="sales ai api")

@app.get("/health")
def health_check():
    return {
        "status":"ok",
        "openai_key_present":bool(settings.OPENAI_API_KEY),
        "weavite_api_key":bool(settings.WEAVIATE_API_KEY),
        "weavite_url":settings.WEAVIATE_URL
    }

app.include_router(search_router.router)