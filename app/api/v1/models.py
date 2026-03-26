from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_ollama_service
from app.services.ollama import OllamaService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", summary="List available Ollama generation models")
async def list_models(
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> dict:
    return {
        "models": await ollama_service.list_models(),
        "default": settings.ollama_model,
    }
