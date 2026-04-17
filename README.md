# Geo-WorldEngine Beta

Geo-WorldEngine是一个（设想中）AI驱动程序化幻想地图生成系统,融合LLM语义理解、计算几何、物理模拟和分布式任务调度,支持自然语言约束的交互式世界构建。

Geo-WorldEngine Beta 是当前的世界生成实验工作区。它包含：

- FastAPI 后端，统一使用 `/api/maps/{taskId}` 任务资源模型
- Vue 3 + Vite 前端，负责提交 prompt、轮询状态、展示预览图与交互地图
- 本地开发模式
- 容器化部署基线
- 新的 RAG 解析增强模块，用于改进 `自然语言 -> WorldPlan`

## 安全

仓库已按公开版本控制做了保守处理：
- `.env`、本地密钥、数据库、产物目录、日志、缓存、虚拟环境默认忽略


## 目录

- `backend/`: FastAPI、任务编排、workers、RAG、存储适配器
- `frontend/Geo-world/`: 前端
- `deployment/`: Compose 与辅助脚本
- `start-beta.ps1`: 本地开发启动脚本
- `.env.example`: 环境变量模板

## 本地开发

前置：

- Python 3.13
- Node.js + npm

启动：

```powershell
Set-Location .\GEO
.\start-beta.ps1
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`

## RAG 模块

beta 现在集成了 P0 版本的只读 RAG：

- 只增强 `自然语言 -> WorldPlan`,不替代地形生成器
- 启动时自动初始化内置 recipe 库
- planner 会记录 `rag_meta`
- 前端任务状态面板会显示：
  - 是否启用 RAG
  - 命中的示例数
  - 最高相似度
  - 回退原因

相关代码：

- `backend/app/rag/`
- `backend/tests/fixtures/rag_eval_set.json`
- `backend/scripts/calibrate_rag_thresholds.py`

### 环境变量

`.env.example` 里新增了：

- `ENABLE_RAG`
- `RECIPE_DB_URL`
- `RAG_MIN_SIMILARITY`
- `RAG_SECOND_DIFF_THRESHOLD`
- `RAG_TOP_K`
- `RAG_LOG_LEVEL`

### 初始化内置知识库

通常在后端启动时自动执行。如果需要手动初始化：

```powershell
Set-Location .\backend
..\.venv313\Scripts\python.exe -m app.rag.init_kb
```

如果上面路径不方便，直接用你本机的 Python 3.13 虚拟环境执行同一命令即可。

### 阈值校准

RAG 默认阈值只是起点，不应当视为最终值。当前仓库已附带一个最小评测集和校准脚本：

```powershell
Set-Location .\backend
..\.venv313\Scripts\python.exe .\scripts\calibrate_rag_thresholds.py
```

输出会给出：

- 相似度分布
- Top-1 / Top-2 差值分布
- 推荐的 `RAG_MIN_SIMILARITY`

当前随仓库提交的默认值已经按这套最小评测集校过一轮：

- `RAG_MIN_SIMILARITY=0.50`
- `RAG_SECOND_DIFF_THRESHOLD=0.10`

## 当前范围

已具备：

- 任务创建与轮询
- 静态预览图生成
- 交互瓦片 manifest
- RAG 增强的 WorldPlan 解析
- 任务资源中的解析诊断信息

仍待继续：

- 扩充 recipe 库
- 基于真实评测集校准阈值
- 将 RAG 命中的结构化 plan 更深地接入底层大陆骨架生成
- 改善地图形态真实性

## 容器化部署

仓库保留了 `postgres + redis + minio + celery + nginx` 的部署基线。  
如果要用容器部署，先复制 `.env.example` 到 `.env`，再用 `deployment/` 下的 compose 文件启动。
