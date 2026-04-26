<script setup lang="ts">
import { computed, ref } from 'vue'

import InteractiveMap from './components/InteractiveMap.vue'
import MapPreview from './components/MapPreview.vue'
import ProgressBar from './components/ProgressBar.vue'
import StageIndicator from './components/StageIndicator.vue'
import TaskForm, { type TaskFormPayload } from './components/TaskForm.vue'
import { createMap, confirmMap } from './api/maps'
import { useTaskPolling } from './composables/useTaskPolling'

const taskId = ref<string | null>(null)
const showInteractive = ref(false)
const submitError = ref<string | null>(null)
const isSubmitting = ref(false)

const { task, isPolling, startPolling, refresh } = useTaskPolling(taskId)

const canShowPreview = computed(() => Boolean(task.value?.previewUrl))
const canShowInteractive = computed(() => Boolean(task.value?.manifestUrl))
const shouldShowProcessing = computed(() => !task.value || task.value.status === 'processing')
const showMapPlaceholder = computed(() => !canShowPreview.value && !showInteractive.value)
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

async function handleSubmit(payload: TaskFormPayload) {
  isSubmitting.value = true
  submitError.value = null
  showInteractive.value = false
  try {
    const created = await createMap(payload)
    taskId.value = created.taskId
    task.value = created
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
  if (!taskId.value) {
    return
  }
  try {
    await confirmMap(taskId.value)
    await refresh()
    startPolling()
  } catch (error) {
    submitError.value = error instanceof Error ? error.message : '确认任务失败'
  }
}

function toggleInteractive() {
  if (!canShowInteractive.value) {
    return
  }
  showInteractive.value = !showInteractive.value
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
      <aside class="sidebar">
        <TaskForm :submitting="isSubmitting" @submit="handleSubmit" />

        <StageIndicator
          :task="task"
          :is-polling="isPolling"
          :submit-error="submitError"
          @confirm="handleConfirm"
        />

        <div class="sidebar-actions">
          <button
            class="btn-outline"
            type="button"
            :disabled="!taskId"
            @click="refresh"
          >
            ↻ 刷新
          </button>
          <button
            class="btn-outline"
            type="button"
            :disabled="!canShowInteractive"
            @click="toggleInteractive"
          >
            {{ showInteractive ? '◁ 静态预览' : '▷ 交互地图' }}
          </button>
        </div>
      </aside>

      <section class="canvas-area">
        <div class="canvas-header">
          <h2>{{ showInteractive && canShowInteractive ? '交互瓦片地图' : canShowPreview ? '世界预览' : '世界画布' }}</h2>
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
            v-show="!showInteractive || !canShowInteractive"
            :preview-url="task?.previewUrl ?? null"
            :visible="canShowPreview"
          />

          <InteractiveMap
            v-show="showInteractive && canShowInteractive"
            :manifest-url="task?.manifestUrl ?? null"
            :active="showInteractive && canShowInteractive"
          />
        </div>
      </section>
    </section>
  </div>
</template>
