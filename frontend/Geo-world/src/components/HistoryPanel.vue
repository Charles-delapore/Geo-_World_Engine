<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getHistory, revertToVersion } from '../api/maps'
import type { VersionInfo } from '../api/types'

const props = defineProps<{
  taskId: string | null
}>()

const emit = defineEmits<{
  (e: 'reverted', versionNum: number): void
}>()

const versions = ref<VersionInfo[]>([])
const isLoading = ref(false)
const reverting = ref(false)

async function loadHistory() {
  if (!props.taskId) return
  isLoading.value = true
  try {
    versions.value = await getHistory(props.taskId)
  } catch {
    versions.value = []
  } finally {
    isLoading.value = false
  }
}

async function revert(versionNum: number) {
  if (!props.taskId || reverting.value) return
  reverting.value = true
  try {
    await revertToVersion(props.taskId, versionNum)
    emit('reverted', versionNum)
    await loadHistory()
  } catch (err) {
    console.error('Revert failed:', err)
  } finally {
    reverting.value = false
  }
}

function formatTime(iso: string) {
  const d = new Date(iso)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

onMounted(() => {
  loadHistory()
})
</script>

<template>
  <div class="history-panel">
    <div class="panel-header">
      <h3 class="panel-title">版本历史</h3>
      <button class="refresh-btn" @click="loadHistory" :disabled="isLoading">刷新</button>
    </div>
    <div v-if="versions.length === 0" class="empty-state">
      暂无编辑历史
    </div>
    <div v-else class="version-list">
      <div
        v-for="v in [...versions].reverse()"
        :key="v.versionNum"
        class="version-item"
      >
        <div class="version-info">
          <span class="version-num">v{{ v.versionNum }}</span>
          <span class="version-summary">{{ v.editSummary || '初始生成' }}</span>
          <span class="version-time">{{ formatTime(v.createdAt) }}</span>
        </div>
        <button
          class="revert-btn"
          :disabled="reverting"
          @click="revert(v.versionNum)"
        >
          回退
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.history-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.panel-title {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  margin: 0;
}
.refresh-btn {
  padding: 3px 8px;
  font-size: 11px;
  border: 1px solid #d9d9d9;
  border-radius: 2px;
  background: #fff;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}
.refresh-btn:hover:not(:disabled) {
  color: #3385ff;
  border-color: #3385ff;
}
.empty-state {
  font-size: 12px;
  color: #999;
  text-align: center;
  padding: 16px;
}
.version-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.version-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 8px;
  background: #f2f3f5;
  border-radius: 2px;
  border: 1px solid #e8e8e8;
}
.version-info {
  display: flex;
  gap: 6px;
  align-items: center;
  font-size: 12px;
}
.version-num {
  color: #3385ff;
  font-weight: 600;
  min-width: 28px;
}
.version-summary {
  color: #666;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 120px;
}
.version-time {
  color: #999;
  font-size: 11px;
}
.revert-btn {
  padding: 2px 8px;
  font-size: 11px;
  border: 1px solid #d9d9d9;
  border-radius: 2px;
  background: transparent;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}
.revert-btn:hover:not(:disabled) {
  color: #3385ff;
  border-color: #3385ff;
}
.revert-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
