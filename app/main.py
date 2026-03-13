import asyncio

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from redis.asyncio import Redis
from sqlalchemy import text

from app.config import settings
from app.db.session import AsyncSessionLocal

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
        allow_origins=settings.cors_allow_origins,
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
    async def health_check() -> JSONResponse:
        checks: dict[str, str] = {}

        try:
            async with AsyncSessionLocal() as session:
                await asyncio.wait_for(
                    session.execute(text("SELECT 1")), timeout=2.0
                )
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {exc}"

        try:
            redis = Redis.from_url(settings.redis_url)
            await asyncio.wait_for(redis.ping(), timeout=2.0)
            await redis.aclose()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"

        healthy = all(v == "ok" for v in checks.values())
        status_code = 200 if healthy else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ok" if healthy else "degraded",
                "version": "0.1.0",
                "env": settings.app_env,
                "checks": checks,
            },
        )

    logger.info("app_created", env=settings.app_env)
    return app


app = create_app()
