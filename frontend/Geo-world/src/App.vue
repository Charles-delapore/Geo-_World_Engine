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

async function handleSubmit(payload: TaskFormPayload) {
  isSubmitting.value = true
  submitError.value = null
  showInteractive.value = false
  try {
    const created = await createMap(payload)
    taskId.value = created.taskId
    task.value = created
    startPolling()
  } catch (error) {
    submitError.value = error instanceof Error ? error.message : '创建任务失败'
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
    <section class="hero-panel">
      <div class="hero-copy">
        <p class="eyebrow">Geo-WorldEngine Beta</p>
        <h1>单一任务资源驱动的世界生成前台</h1>
        <p class="hero-text">
          默认展示静态预览图，交互瓦片只在产物就绪后由用户主动切换。当前这版已经接入 beta 后端任务状态机与预览链路。
        </p>
      </div>
      <TaskForm :submitting="isSubmitting" @submit="handleSubmit" />
    </section>

    <section class="workspace-grid">
      <aside class="status-panel">
        <StageIndicator
          :task="task"
          :is-polling="isPolling"
          :submit-error="submitError"
          @confirm="handleConfirm"
        />
        <ProgressBar :value="task?.progress ?? 0" />
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
          <div v-if="shouldShowProcessing" class="placeholder-card">
            <p>提交任务后，这里会首先展示生成进度，再展示首屏预览图。</p>
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
