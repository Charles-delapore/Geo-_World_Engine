<script setup lang="ts">
defineProps<{
  value: number
}>()
</script>

<template>
  <div class="progress-shell" aria-label="任务进度">
    <div class="progress-track">
      <div class="progress-fill" :style="{ width: `${Math.max(0, Math.min(100, value))}%` }" />
    </div>
    <span class="progress-label">{{ value }}%</span>
  </div>
</template>
