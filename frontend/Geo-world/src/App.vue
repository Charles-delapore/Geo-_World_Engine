<script setup lang="ts">
import { computed, ref } from 'vue'

import CesiumGlobe from './components/CesiumGlobe.vue'
import EditPanel from './components/EditPanel.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import InteractiveMap from './components/InteractiveMap.vue'
import MapControls from './components/MapControls.vue'
import MapPreview from './components/MapPreview.vue'
import ProgressBar from './components/ProgressBar.vue'
import SketchCanvas from './components/SketchCanvas.vue'
import StageIndicator from './components/StageIndicator.vue'
import TaskForm, { type TaskFormPayload } from './components/TaskForm.vue'
import { applySketch, createMap, confirmMap } from './api/maps'
import { useTaskPolling } from './composables/useTaskPolling'

const taskId = ref<string | null>(null)
const showInteractive = ref(false)
const submitError = ref<string | null>(null)
const isSubmitting = ref(false)
const currentProjection = ref<'flat' | 'planet'>('planet')
const brushActive = ref(false)
const brushApplying = ref(false)
const editTimestamp = ref(0)
const currentBrushMode = ref<'none' | 'raise' | 'lower' | 'flatten' | 'roughen'>('none')
const currentBrushSize = ref(5)
const activeTab = ref<'create' | 'status' | 'edit' | 'history'>('create')
const sidebarOpen = ref(true)

const mapZoom = ref(0)
const mapCenter = ref<[number, number] | null>(null)
const mapMinZoom = ref(0)
const mapMaxZoom = ref(6)

const interactiveMapRef = ref<InstanceType<typeof InteractiveMap> | null>(null)
const globeRef = ref<InstanceType<typeof CesiumGlobe> | null>(null)

const { task, isPolling, pollError, startPolling, refresh } = useTaskPolling(taskId)

const canShowPreview = computed(() => Boolean(task.value?.previewUrl))
const canShowInteractive = computed(() => Boolean(task.value?.manifestUrl || task.value?.previewUrl))
const shouldShowProcessing = computed(() => !task.value || task.value.status === 'processing')
const showMapPlaceholder = computed(() => !canShowPreview.value && !showInteractive.value)
const isEditable = computed(() => task.value?.status === 'ready-interactive' || task.value?.status === 'ready-image')
const statusLabel = computed(() => {
  if (!task.value) return '空闲'
  const s = task.value.status
  if (s === 'processing') return '生成中'
  if (s === 'awaiting-confirm') return '待确认'
  if (s === 'ready-image') return '预览就绪'
  if (s === 'ready-interactive') return '交互就绪'
  if (s === 'failed') return '失败'
  return s
})

const sidebarTabs = computed(() => {
  const tabs: Array<{ key: 'create' | 'status' | 'edit' | 'history'; icon: string; label: string }> = [
    { key: 'create' as const, icon: '✦', label: '创建' },
    { key: 'status' as const, icon: '◎', label: '状态' },
  ]
  if (isEditable.value) {
    tabs.push({ key: 'edit' as const, icon: '✎', label: '编辑' })
    tabs.push({ key: 'history' as const, icon: '⏎', label: '历史' })
  }
  return tabs
})

async function handleSubmit(payload: TaskFormPayload) {
  isSubmitting.value = true
  submitError.value = null
  showInteractive.value = false
  try {
    const created = await createMap(payload)
    taskId.value = created.taskId
    task.value = created
    currentProjection.value = payload.projection
    activeTab.value = 'status'
    startPolling()
  } catch (error: unknown) {
    if (error && typeof error === 'object' && 'response' in error) {
      const axiosError = error as { response?: { status?: number; data?: unknown }; message?: string }
      const status = axiosError.response?.status
      const data = axiosError.response?.data
      submitError.value = `请求失败 (${status || '未知'}): ${typeof data === 'object' ? JSON.stringify(data) : String(data || axiosError.message || '未知错误')}`
    } else {
      submitError.value = error instanceof Error ? error.message : '创建任务失败'
    }
  } finally {
    isSubmitting.value = false
  }
}

async function handleConfirm() {
  if (!taskId.value) return
  try {
    await confirmMap(taskId.value)
    await refresh()
    startPolling()
  } catch (error) {
    submitError.value = error instanceof Error ? error.message : '确认任务失败'
  }
}

function toggleInteractive() {
  if (!canShowInteractive.value) return
  showInteractive.value = !showInteractive.value
}

function handleEdited() {
  editTimestamp.value = Date.now()
  refresh()
  startPolling()
}

function handleReverted() {
  editTimestamp.value = Date.now()
  refresh()
  startPolling()
}

function handleBrushModeChanged(mode: 'none' | 'raise' | 'lower' | 'flatten' | 'roughen', size?: number) {
  currentBrushMode.value = mode
  brushActive.value = mode !== 'none'
  if (size !== undefined) {
    currentBrushSize.value = size
  }
}

async function handleStrokeComplete(geojson: Record<string, unknown>) {
  if (!taskId.value || currentBrushMode.value === 'none' || brushApplying.value) return
  try {
    brushApplying.value = true
    await applySketch(taskId.value, geojson, currentBrushMode.value, currentBrushSize.value)
    editTimestamp.value = Date.now()
    await refresh()
    startPolling()
  } catch (err) {
    console.error('Brush stroke application failed:', err)
    submitError.value = '画笔编辑应用失败'
  } finally {
    brushApplying.value = false
  }
}

function handleZoomChange(zoom: number) {
  mapZoom.value = zoom
}

function handleCenterChange(center: [number, number]) {
  mapCenter.value = center
}

function handleZoomRange(min: number, max: number) {
  mapMinZoom.value = min
  mapMaxZoom.value = max
}

function handleZoomIn() {
  if (currentProjection.value === 'planet') {
    globeRef.value?.zoomIn()
    return
  }
  interactiveMapRef.value?.zoomIn()
}

function handleZoomOut() {
  if (currentProjection.value === 'planet') {
    globeRef.value?.zoomOut()
    return
  }
  interactiveMapRef.value?.zoomOut()
}
</script>

<template>
  <div class="app-shell">
    <header class="top-bar">
      <div class="brand">
        <span class="brand-icon">◆</span>
        <div>
          <h1 class="brand-title">Geo-WorldEngine</h1>
          <p class="brand-sub">Beta · 程序化世界生成</p>
        </div>
      </div>
      <div v-if="taskId" class="status-badge" :class="task?.status">
        <span class="status-dot" />
        {{ statusLabel }}
      </div>
    </header>

    <section class="main-layout">
      <aside class="sidebar" :class="{ 'sidebar-collapsed': !sidebarOpen }">
        <nav class="sidebar-nav">
          <button
            v-for="tab in sidebarTabs"
            :key="tab.key"
            class="nav-item"
            :class="{ active: activeTab === tab.key }"
            :title="tab.label"
            @click="activeTab = tab.key"
          >
            <span class="nav-icon">{{ tab.icon }}</span>
            <span v-show="sidebarOpen" class="nav-label">{{ tab.label }}</span>
          </button>

          <div class="nav-spacer" />

          <button class="nav-item toggle-btn" :title="sidebarOpen ? '收起侧边栏' : '展开侧边栏'" @click="sidebarOpen = !sidebarOpen">
            <svg class="nav-icon" viewBox="0 0 24 24" width="16" height="16">
              <template v-if="sidebarOpen">
                <polyline points="15 18 9 12 15 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </template>
              <template v-else>
                <polyline points="9 18 15 12 9 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </template>
            </svg>
            <span v-show="sidebarOpen" class="nav-label">收起</span>
          </button>
        </nav>

        <div v-show="sidebarOpen" class="sidebar-body">
          <div v-show="activeTab === 'create'" class="tab-panel">
            <TaskForm :submitting="isSubmitting" @submit="handleSubmit" />
          </div>

          <div v-show="activeTab === 'status'" class="tab-panel">
            <StageIndicator
              :task="task"
              :is-polling="isPolling"
              :submit-error="submitError"
              :poll-error="pollError"
              @confirm="handleConfirm"
            />
            <div class="sidebar-actions">
              <button class="btn-outline" type="button" :disabled="!taskId" @click="refresh">↻ 刷新</button>
              <button class="btn-outline" type="button" :disabled="!canShowInteractive" @click="toggleInteractive">
                {{ showInteractive ? '◁ 静态预览' : '▷ 交互地图' }}
              </button>
            </div>
          </div>

          <div v-if="activeTab === 'edit' && isEditable" class="tab-panel">
            <EditPanel
              :task-id="taskId"
              @edited="handleEdited"
              @brush-mode-changed="handleBrushModeChanged"
            />
          </div>

          <div v-if="activeTab === 'history' && isEditable" class="tab-panel">
            <HistoryPanel
              :task-id="taskId"
              @reverted="handleReverted"
            />
          </div>
        </div>
      </aside>

      <section class="canvas-area">
        <div class="canvas-header">
          <h2>{{ showInteractive && canShowInteractive ? (currentProjection === 'planet' ? '星球视图' : '交互瓦片地图') : canShowPreview ? '世界预览' : '世界画布' }}</h2>
          <span v-if="taskId" class="task-tag">{{ taskId.slice(0, 8) }}</span>
        </div>

        <div class="canvas-body">
          <div v-if="shouldShowProcessing" class="canvas-overlay">
            <div class="overlay-content">
              <div class="spinner" />
              <p class="overlay-stage">{{ task?.currentStage ?? '等待新任务' }}</p>
              <p class="overlay-hint">
                {{ task ? '世界正在生成中…' : '输入描述后点击生成，世界将在此呈现。' }}
              </p>
              <ProgressBar v-if="task" :value="task.progress ?? 0" />
            </div>
          </div>

          <div v-if="showMapPlaceholder && !shouldShowProcessing" class="canvas-empty">
            <div class="empty-icon">🗺</div>
            <p>等待世界生成</p>
          </div>

          <MapPreview
            v-show="canShowPreview && !showInteractive"
            :preview-url="task?.previewUrl ?? null"
            :visible="canShowPreview && !showInteractive"
          />

          <InteractiveMap
            v-if="showInteractive && canShowInteractive && currentProjection === 'flat'"
            ref="interactiveMapRef"
            :manifest-url="task?.manifestUrl ?? null"
            :preview-url="task?.previewUrl ?? null"
            :active="showInteractive && canShowInteractive"
            projection="flat"
            @zoom-change="handleZoomChange"
            @center-change="handleCenterChange"
            @zoom-range="handleZoomRange"
          />

          <CesiumGlobe
            v-if="showInteractive && canShowInteractive && currentProjection === 'planet'"
            ref="globeRef"
            :manifest-url="task?.manifestUrl ?? null"
            :preview-url="task?.previewUrl ?? null"
            :active="showInteractive && canShowInteractive"
            :edit-timestamp="editTimestamp"
            @center-change="handleCenterChange"
            @zoom-change="handleZoomChange"
            @zoom-range="handleZoomRange"
          />

          <SketchCanvas
            v-if="showInteractive && canShowInteractive && brushActive"
            :active="brushActive"
            :brush-mode="(currentBrushMode === 'none' ? 'raise' : currentBrushMode) as 'raise' | 'lower' | 'flatten' | 'roughen'"
            :brush-size="currentBrushSize"
            @stroke-complete="handleStrokeComplete"
          />

          <MapControls
            v-if="showInteractive && canShowInteractive"
            :zoom="mapZoom"
            :min-zoom="mapMinZoom"
            :max-zoom="mapMaxZoom"
            :center="mapCenter"
            :projection="currentProjection"
            :show-interactive="showInteractive"
            :can-show-interactive="canShowInteractive"
            @zoom-in="handleZoomIn"
            @zoom-out="handleZoomOut"
            @toggle-view="toggleInteractive"
          />
        </div>
      </section>
    </section>
  </div>
</template>
