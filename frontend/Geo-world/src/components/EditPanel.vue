<script setup lang="ts">
import { ref, computed } from 'vue'
import { editMap } from '../api/maps'
import type { EditResponse } from '../api/types'

const props = defineProps<{
  taskId: string | null
}>()

const emit = defineEmits<{
  (e: 'edited', response: EditResponse): void
  (e: 'brush-mode-changed', mode: 'none' | 'raise' | 'lower' | 'flatten' | 'roughen', size?: number): void
}>()

const editForm = ref<'text' | 'brush'>('text')
const instruction = ref('')
const brushMode = ref<'none' | 'raise' | 'lower' | 'flatten' | 'roughen'>('none')
const brushSize = ref(5)
const isSubmitting = ref(false)

const brushActive = computed(() => editForm.value === 'brush' && brushMode.value !== 'none')

function switchToText() {
  editForm.value = 'text'
  emit('brush-mode-changed', 'none')
}

function switchToBrush() {
  editForm.value = 'brush'
  emit('brush-mode-changed', brushMode.value, brushSize.value)
}

function changeBrushMode(mode: 'none' | 'raise' | 'lower' | 'flatten' | 'roughen') {
  brushMode.value = mode
  emit('brush-mode-changed', mode, brushSize.value)
}

function changeBrushSize() {
  if (brushActive.value) {
    emit('brush-mode-changed', brushMode.value, brushSize.value)
  }
}

async function submitTextEdit() {
  if (!props.taskId || !instruction.value.trim()) return
  isSubmitting.value = true
  try {
    const response = await editMap(props.taskId, { text_instruction: instruction.value })
    instruction.value = ''
    emit('edited', response)
  } catch (err) {
    console.error('Edit failed:', err)
  } finally {
    isSubmitting.value = false
  }
}
</script>

<template>
  <div class="edit-panel">
    <h3 class="panel-title">增量编辑</h3>
    <div class="edit-form-selector">
      <button :class="{ active: editForm === 'text' }" @click="switchToText">📝 文本</button>
      <button :class="{ active: editForm === 'brush' }" @click="switchToBrush">🖌 画笔</button>
    </div>

    <div v-if="editForm === 'text'" class="text-edit-section">
      <textarea
        v-model="instruction"
        class="edit-input"
        placeholder="输入编辑指令，如：在东侧添加一条南北向山脉"
        rows="3"
        :disabled="!taskId || isSubmitting"
      />
      <button
        class="submit-btn"
        :disabled="!taskId || !instruction.trim() || isSubmitting"
        @click="submitTextEdit"
      >
        {{ isSubmitting ? '处理中...' : '应用修改' }}
      </button>
    </div>

    <div v-if="editForm === 'brush'" class="brush-edit-section">
      <div class="brush-mode-selector">
        <button
          :class="{ active: brushMode === 'none' }"
          @click="changeBrushMode('none')"
          title="移动地图"
        >
          ✋ 移动
        </button>
        <button
          :class="{ active: brushMode === 'raise' }"
          @click="changeBrushMode('raise')"
          title="抬升地形"
        >
          ⬆ 抬升
        </button>
        <button
          :class="{ active: brushMode === 'lower' }"
          @click="changeBrushMode('lower')"
          title="降低地形"
        >
          ⬇ 降低
        </button>
        <button
          :class="{ active: brushMode === 'flatten' }"
          @click="changeBrushMode('flatten')"
          title="填平为水面"
        >
          ≈ 填平
        </button>
        <button
          :class="{ active: brushMode === 'roughen' }"
          @click="changeBrushMode('roughen')"
          title="增加粗糙度"
        >
          ≋ 粗糙
        </button>
      </div>
      <div class="brush-size-control">
        <span class="size-label">笔刷大小</span>
        <input
          v-model.number="brushSize"
          type="range"
          min="1"
          max="15"
          step="1"
          class="size-slider"
          @input="changeBrushSize"
        />
        <span class="size-value">{{ brushSize }}</span>
      </div>
      <p class="brush-hint">
        在地图上拖动绘制，松开后自动应用
      </p>
    </div>
  </div>
</template>

<style scoped>
.edit-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.panel-title {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  margin: 0;
}
.edit-form-selector {
  display: flex;
  gap: 4px;
}
.edit-form-selector button {
  flex: 1;
  padding: 7px 12px;
  font-size: 13px;
  border: 1px solid #d9d9d9;
  border-radius: 2px;
  background: #fff;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}
.edit-form-selector button.active {
  background: #3385ff;
  color: #fff;
  border-color: #3385ff;
}
.text-edit-section {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.edit-input {
  width: 100%;
  padding: 8px 10px;
  font-size: 13px;
  border: 1px solid #d9d9d9;
  border-radius: 2px;
  background: #fff;
  color: #333;
  resize: vertical;
  font-family: inherit;
}
.edit-input:focus {
  outline: none;
  border-color: #3385ff;
  box-shadow: 0 0 0 2px rgba(51, 133, 255, 0.1);
}
.submit-btn {
  padding: 7px 16px;
  font-size: 13px;
  border: none;
  border-radius: 2px;
  background: #3385ff;
  color: #fff;
  cursor: pointer;
  transition: background 0.2s;
}
.submit-btn:hover:not(:disabled) {
  background: #5a9fff;
}
.submit-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.brush-edit-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.brush-mode-selector {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.brush-mode-selector button {
  padding: 5px 10px;
  font-size: 12px;
  border: 1px solid #d9d9d9;
  border-radius: 2px;
  background: #fff;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}
.brush-mode-selector button.active {
  background: #3385ff;
  color: #fff;
  border-color: #3385ff;
}
.brush-size-control {
  display: flex;
  align-items: center;
  gap: 8px;
}
.size-label {
  font-size: 12px;
  color: #666;
  white-space: nowrap;
}
.size-slider {
  flex: 1;
  accent-color: #3385ff;
}
.size-value {
  font-size: 12px;
  color: #333;
  min-width: 20px;
  text-align: center;
}
.brush-hint {
  font-size: 11px;
  color: #999;
  margin: 0;
}
</style>
