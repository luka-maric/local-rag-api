import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings

logger = structlog.get_logger()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-tenant RAG API",
        description="Document Q&A with tenant isolation",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],     # tighten this in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        Instrumentator(should_group_status_codes=False).instrument(app).expose(app)
    except ValueError:
        pass  # already registered — safe to ignore

    from app.api.v1.auth import router as auth_router
    from app.api.v1.chat import router as chat_router
    from app.api.v1.documents import router as documents_router
    from app.api.v1.query import router as query_router
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")

    @app.get("/health", tags=["infrastructure"])
    async def health_check():
        """Lightweight liveness check — no database calls."""
        return {
            "status": "ok",
            "env": settings.app_env,
            "version": "0.1.0",
        }

    logger.info("app_created", env=settings.app_env)
    return app


app = create_app()
