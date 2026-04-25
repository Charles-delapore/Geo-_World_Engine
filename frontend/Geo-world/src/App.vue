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
    <section class="control-deck">
      <div class="hero-copy">
        <p class="eyebrow">Geo-WorldEngine Beta</p>
        <h1>单一任务资源驱动的世界生成前台</h1>
        <p class="hero-text">
          默认展示静态预览图，交互瓦片只在产物就绪后由用户主动切换。上半区合并为单一控制台，当前任务的约束摘要和模拟参数会在下方持续刷新。
        </p>
      </div>
      <TaskForm class="control-form" :submitting="isSubmitting" @submit="handleSubmit" />
    </section>

    <section class="workspace-grid">
      <aside class="status-panel">
        <StageIndicator
          :task="task"
          :is-polling="isPolling"
          :submit-error="submitError"
          @confirm="handleConfirm"
        />
        <div class="status-actions">
          <button
            class="secondary-button"
            type="button"
            :disabled="!taskId"
            @click="refresh"
          >
            刷新状态
          </button>
          <button
            class="secondary-button"
            type="button"
            :disabled="!canShowInteractive"
            @click="toggleInteractive"
          >
            {{ showInteractive ? '返回静态预览' : '切换到交互地图' }}
          </button>
        </div>
      </aside>

      <section class="artifact-panel">
        <header class="artifact-header">
          <div>
            <p class="artifact-kicker">产物视图</p>
            <h2>{{ showInteractive && canShowInteractive ? '交互瓦片地图' : '静态预览图' }}</h2>
          </div>
          <span v-if="taskId" class="task-chip">Task {{ taskId }}</span>
        </header>

        <div class="artifact-body">
          <div v-if="shouldShowProcessing" class="artifact-overlay">
            <p class="overlay-title">{{ task?.currentStage ?? '等待新任务' }}</p>
            <p class="overlay-copy">
              {{ task ? '生成进行中，预览图会在首个产物完成后自动替换这里。' : '提交任务后，这里会直接显示地图区域中的生成进度。' }}
            </p>
            <ProgressBar :value="task?.progress ?? 0" />
          </div>

          <div v-if="showMapPlaceholder" class="map-placeholder">
            <p>预览图尚未生成。</p>
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
