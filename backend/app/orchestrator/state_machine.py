from __future__ import annotations

from app.storage.models import TaskStatus


STAGE_LABELS = {
    TaskStatus.QUEUED: "任务已排队",
    TaskStatus.PARSING: "正在解析世界约束",
    TaskStatus.AWAITING_CONFIRM: "等待用户确认世界规划",
    TaskStatus.GENERATING_TERRAIN: "正在生成地形与侵蚀",
    TaskStatus.RENDERING_IMAGE: "正在渲染预览图",
    TaskStatus.READY: "静态预览已就绪",
    TaskStatus.READY_INTERACTIVE: "交互地图已就绪",
    TaskStatus.FAILED: "任务执行失败",
}


PUBLIC_STATUS = {
    TaskStatus.QUEUED: "processing",
    TaskStatus.PARSING: "processing",
    TaskStatus.AWAITING_CONFIRM: "awaiting-confirm",
    TaskStatus.GENERATING_TERRAIN: "processing",
    TaskStatus.RENDERING_IMAGE: "processing",
    TaskStatus.READY: "ready-image",
    TaskStatus.READY_INTERACTIVE: "ready-interactive",
    TaskStatus.FAILED: "failed",
}
