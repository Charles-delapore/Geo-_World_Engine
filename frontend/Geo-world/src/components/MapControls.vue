<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  zoom: number
  minZoom: number
  maxZoom: number
  center: [number, number] | null
  projection: 'flat' | 'planet'
  showInteractive: boolean
  canShowInteractive: boolean
}>()

const emit = defineEmits<{
  (e: 'zoom-in'): void
  (e: 'zoom-out'): void
  (e: 'toggle-view'): void
}>()

const coordText = computed(() => {
  if (!props.center) return '--°, --°'
  const [lng, lat] = props.center
  const lngDir = lng >= 0 ? 'E' : 'W'
  const latDir = lat >= 0 ? 'N' : 'S'
  return `${Math.abs(lng).toFixed(2)}°${lngDir}  ${Math.abs(lat).toFixed(2)}°${latDir}`
})

const projectionLabel = computed(() => {
  return props.projection === 'planet' ? '星球' : '平面'
})
</script>

<template>
  <div class="map-controls">
    <div class="controls-right">
      <div class="zoom-control">
        <button class="ctrl-btn zoom-in" title="放大" :disabled="zoom >= maxZoom" @click="$emit('zoom-in')">
          <svg viewBox="0 0 24 24" width="16" height="16"><line x1="12" y1="5" x2="12" y2="19" stroke="currentColor" stroke-width="2"/><line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" stroke-width="2"/></svg>
        </button>
        <div class="zoom-level" :title="`缩放: ${zoom}`">{{ zoom }}</div>
        <button class="ctrl-btn zoom-out" title="缩小" :disabled="zoom <= minZoom" @click="$emit('zoom-out')">
          <svg viewBox="0 0 24 24" width="16" height="16"><line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" stroke-width="2"/></svg>
        </button>
      </div>

      <button
        v-if="canShowInteractive"
        class="ctrl-btn view-toggle"
        :title="showInteractive ? '切换到静态预览' : '切换到交互地图'"
        @click="$emit('toggle-view')"
      >
        <svg viewBox="0 0 24 24" width="16" height="16">
          <template v-if="showInteractive">
            <rect x="3" y="3" width="18" height="18" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/>
            <line x1="9" y1="3" x2="9" y2="21" stroke="currentColor" stroke-width="1"/>
          </template>
          <template v-else>
            <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.5"/>
            <ellipse cx="12" cy="12" rx="4" ry="9" fill="none" stroke="currentColor" stroke-width="1"/>
            <line x1="3" y1="12" x2="21" y2="12" stroke="currentColor" stroke-width="1"/>
          </template>
        </svg>
      </button>
    </div>

    <div class="controls-bottom">
      <div class="coord-display">
        <svg viewBox="0 0 24 24" width="12" height="12" class="coord-icon"><circle cx="12" cy="12" r="3" fill="currentColor"/><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>
        <span>{{ coordText }}</span>
      </div>
      <div class="projection-badge">
        {{ projectionLabel }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.map-controls {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 10;
}

.controls-right {
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  pointer-events: auto;
}

.zoom-control {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #fff;
  border-radius: 2px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
  overflow: hidden;
}

.ctrl-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: none;
  background: #fff;
  color: #333;
  cursor: pointer;
  transition: color 0.2s, background 0.2s;
}

.ctrl-btn:hover:not(:disabled) {
  color: #3385ff;
  background: #f5f7fa;
}

.ctrl-btn:disabled {
  color: #ccc;
  cursor: not-allowed;
}

.zoom-level {
  width: 32px;
  height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: #666;
  border-top: 1px solid #e8e8e8;
  border-bottom: 1px solid #e8e8e8;
  font-variant-numeric: tabular-nums;
}

.view-toggle {
  background: #fff;
  border-radius: 2px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
}

.controls-bottom {
  position: absolute;
  bottom: 8px;
  left: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
  pointer-events: auto;
}

.coord-display {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 2px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  font-size: 11px;
  color: #666;
  font-variant-numeric: tabular-nums;
  font-family: "SF Mono", "Cascadia Code", "Consolas", monospace;
}

.coord-icon {
  color: #d81e06;
  flex-shrink: 0;
}

.projection-badge {
  padding: 3px 8px;
  background: rgba(51, 133, 255, 0.9);
  border-radius: 2px;
  font-size: 11px;
  color: #fff;
  font-weight: 500;
  letter-spacing: 0.02em;
}
</style>
