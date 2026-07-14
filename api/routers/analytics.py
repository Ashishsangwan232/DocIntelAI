"""
routers/analytics.py
=====================
Read-only usage statistics for the Analytics dashboard. Thin wrapper
around `AnalyticsService`, which is itself already read-only and
side-effect-free.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_analytics_service
from api.schemas.analytics import AnalyticsSummaryResponse
from src.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
def get_analytics_summary(
    most_queried_limit: int = Query(default=5, ge=1, le=50),
    activity_days: int = Query(default=14, ge=1, le=365),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsSummaryResponse:
    summary = analytics_service.get_summary(
        most_queried_limit=most_queried_limit, activity_days=activity_days
    )
    return AnalyticsSummaryResponse.from_summary(summary)
