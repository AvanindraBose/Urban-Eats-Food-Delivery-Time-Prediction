from fastapi import APIRouter, Depends
from backend.loaders.model_pipeline_loader import load_artifacts
from backend.logging_fastapi.logger_api import health_logger

router = APIRouter(
    prefix="/internal",
    tags=["Health"]
)


@router.get("/health")
async def health_check(
    model=Depends(load_artifacts)
):
    health_logger.save_logs(
        "Health check initiated",
        log_level="info"
    )

    try:
        assert model is not None

        health_logger.save_logs(
            "Model loaded successfully",
            log_level="info"
        )

        return {"status": "ok"}

    except Exception as e:
        health_logger.save_logs(
            f"Health check failed: {e}",
            log_level="error"
        )

        return {
            "status": "error",
            "details": str(e)
        }