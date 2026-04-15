<script setup lang="ts">
import type { MapTask } from '../api/types'

defineProps<{
  task: MapTask | null
  isPolling: boolean
  submitError: string | null
}>()

defineEmits<{
  confirm: []
}>()
</script>

<template>
  <div class="stage-panel">
    <p class="artifact-kicker">任务状态</p>
    <h2>{{ task?.currentStage ?? '等待新任务' }}</h2>
    <p class="stage-copy">
      {{ task?.planSummary ?? '前端只消费一个 /maps/{taskId} 资源，页面状态由任务资源本身驱动。' }}
    </p>
    <p v-if="isPolling" class="stage-meta">轮询中，等待后端状态推进。</p>
    <p v-if="task?.errorMsg" class="error-text">{{ task.errorMsg }}</p>
    <p v-if="submitError" class="error-text">{{ submitError }}</p>
    <button
      v-if="task?.status === 'awaiting-confirm'"
      class="primary-button"
      type="button"
      @click="$emit('confirm')"
    >
      确认世界规划并继续
    </button>
  </div>
</template>
