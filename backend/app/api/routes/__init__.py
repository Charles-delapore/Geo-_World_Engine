from fastapi import APIRouter

from app.api.routes.artifacts import router as artifacts_router
from app.api.routes.health import router as health_router
from app.api.routes.internal import router as internal_router
from app.api.routes.maps import router as maps_router

router = APIRouter()
router.include_router(health_router)
router.include_router(maps_router)
router.include_router(artifacts_router)
router.include_router(internal_router)
