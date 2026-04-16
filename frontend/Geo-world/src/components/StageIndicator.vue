<script setup lang="ts">
import { computed } from 'vue'

import type { MapTask } from '../api/types'

const props = defineProps<{
  task: MapTask | null
  isPolling: boolean
  submitError: string | null
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
    { label: '海域类型', value: diagnostics.seaStyle },
    { label: '大陆数', value: String(diagnostics.continentCount) },
    { label: '山脉约束', value: String(diagnostics.mountainCount) },
    { label: '海域约束', value: String(diagnostics.seaZoneCount) },
    { label: '陆地率', value: `${Math.round(diagnostics.landRatio * 100)}%` },
    { label: '崎岖度', value: `${Math.round(diagnostics.ruggedness * 100)}%` },
    { label: '海岸复杂度', value: `${Math.round(diagnostics.coastComplexity * 100)}%` },
    { label: '岛链倾向', value: `${Math.round(diagnostics.islandFactor * 100)}%` },
    { label: '湿润度', value: diagnostics.moisture.toFixed(2) },
    { label: '温度偏置', value: diagnostics.temperatureBias.toFixed(1) },
    { label: '主导风向', value: diagnostics.windDirection },
    { label: 'RAG启用', value: diagnostics.ragEnabled ? 'yes' : 'no' },
    { label: '示例命中', value: String(diagnostics.ragExamples) },
    { label: '最高相似度', value: diagnostics.ragTopSimilarity == null ? '-' : diagnostics.ragTopSimilarity.toFixed(3) },
    { label: '回退原因', value: diagnostics.ragFallbackReason ?? '-' },
    { label: '分辨率', value: `${diagnostics.width}×${diagnostics.height}` },
  ]
})
</script>

<template>
  <div class="stage-panel">
    <p class="artifact-kicker">任务状态</p>
    <h2>{{ task?.currentStage ?? '等待新任务' }}</h2>
    <p class="stage-copy">
      {{ task?.planSummary ?? '这里展示当前世界生成任务的约束摘要与数值迭代，而进度条只保留在产物视图。' }}
    </p>
    <div v-if="metrics.length" class="metrics-grid">
      <div v-for="item in metrics" :key="item.label" class="metric-card">
        <span class="metric-label">{{ item.label }}</span>
        <strong class="metric-value">{{ item.value }}</strong>
      </div>
    </div>
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
