<script setup lang="ts">
import { computed } from 'vue'

import type { MapTask } from '../api/types'

const props = defineProps<{
  task: MapTask | null
  isPolling: boolean
  submitError: string | null
  pollError: string | null
}>()

defineEmits<{
  confirm: []
}>()

const metrics = computed(() => {
  const diagnostics = props.task?.diagnostics
  if (!diagnostics) {
    return []
  }

  return [
    { label: '版型', value: diagnostics.layoutTemplate },
    { label: '海域', value: diagnostics.seaStyle },
    { label: '大陆', value: String(diagnostics.continentCount) },
    { label: '山脉', value: String(diagnostics.mountainCount) },
    { label: '海域约束', value: String(diagnostics.seaZoneCount) },
    { label: '陆地率', value: `${Math.round(diagnostics.landRatio * 100)}%` },
    { label: '崎岖度', value: `${Math.round(diagnostics.ruggedness * 100)}%` },
    { label: '海岸', value: `${Math.round(diagnostics.coastComplexity * 100)}%` },
    { label: '岛链', value: `${Math.round(diagnostics.islandFactor * 100)}%` },
    { label: '湿润度', value: diagnostics.moisture.toFixed(2) },
    { label: '温度', value: diagnostics.temperatureBias.toFixed(1) },
    { label: '风向', value: diagnostics.windDirection },
    { label: 'RAG', value: diagnostics.ragEnabled ? '✓' : '✗' },
    { label: '命中', value: String(diagnostics.ragExamples) },
    { label: '相似度', value: diagnostics.ragTopSimilarity == null ? '-' : diagnostics.ragTopSimilarity.toFixed(3) },
    { label: '回退', value: diagnostics.ragFallbackReason ?? '-' },
    { label: '分辨率', value: `${diagnostics.width}×${diagnostics.height}` },
  ]
})

const pollingText = computed(() => {
  if (!props.isPolling) return ''
  if (props.task?.status === 'ready-image' && props.task?.diagnostics?.generateTiles !== false) {
    return '交互瓦片生成中…'
  }
  return '轮询中…'
})
</script>

<template>
  <div class="stage-panel">
    <h2>{{ task?.currentStage ?? '等待任务' }}</h2>
    <p class="stage-copy">
      {{ task?.planSummary ?? '提交描述后，生成参数和诊断信息将在此展示。' }}
    </p>
    <div v-if="metrics.length" class="metrics-grid">
      <div v-for="item in metrics" :key="item.label" class="metric-card">
        <span class="metric-label">{{ item.label }}</span>
        <strong class="metric-value">{{ item.value }}</strong>
      </div>
    </div>
    <p v-if="pollingText" class="stage-meta">{{ pollingText }}</p>
    <p v-if="task?.errorMsg" class="error-text">{{ task.errorMsg }}</p>
    <p v-if="submitError" class="error-text">{{ submitError }}</p>
    <p v-if="pollError" class="error-text">{{ pollError }}</p>
    <button
      v-if="task?.status === 'awaiting-confirm'"
      class="primary-button"
      type="button"
      @click="$emit('confirm')"
    >
      确认规划并继续
    </button>
  </div>
</template>
