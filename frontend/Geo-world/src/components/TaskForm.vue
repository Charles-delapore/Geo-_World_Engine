<script setup lang="ts">
import { reactive } from 'vue'

export interface TaskFormPayload {
  prompt: string
  width: number
  height: number
  seed?: number
  auto_confirm: boolean
  generate_tiles: boolean
  projection: 'flat' | 'planet'
}

defineProps<{
  submitting: boolean
}>()

const emit = defineEmits<{
  submit: [payload: TaskFormPayload]
}>()

const form = reactive({
  prompt: '一东一西两块大陆，中间有内海，北部有山脉',
  width: 1024,
  height: 512,
  seed: '',
  auto_confirm: true,
  generate_tiles: true,
  projection: 'planet' as 'flat' | 'planet',
})

function onSubmit() {
  emit('submit', {
    prompt: form.prompt,
    width: form.width,
    height: form.height,
    seed: form.seed ? Number(form.seed) : undefined,
    auto_confirm: form.auto_confirm,
    generate_tiles: form.generate_tiles,
    projection: form.projection,
  })
}
</script>

<template>
  <form class="task-form" @submit.prevent="onSubmit">
    <label>
      <span>世界描述</span>
      <textarea v-model="form.prompt" rows="5" placeholder="描述大陆、山脉、内海和整体风格" />
    </label>

    <div class="task-grid">
      <label>
        <span>宽度</span>
        <input v-model.number="form.width" type="number" min="256" max="2048" step="256" />
      </label>
      <label>
        <span>高度</span>
        <input v-model.number="form.height" type="number" min="256" max="1024" step="128" />
      </label>
      <label>
        <span>种子</span>
        <input v-model="form.seed" type="number" min="1" placeholder="留空则自动生成" />
      </label>
    </div>

    <div class="projection-selector">
      <span class="selector-label">投影模式</span>
      <div class="selector-buttons">
        <button
          type="button"
          :class="{ active: form.projection === 'planet' }"
          @click="form.projection = 'planet'"
        >
          🌐 星球
        </button>
        <button
          type="button"
          :class="{ active: form.projection === 'flat' }"
          @click="form.projection = 'flat'"
        >
          🗺 平面
        </button>
      </div>
      <p class="selector-hint">
        {{ form.projection === 'planet' ? 'X轴首尾衔接，可无限横向移动' : '有边界的平面坐标系，限制在地图范围内' }}
      </p>
    </div>

    <label class="checkbox-row">
      <input v-model="form.auto_confirm" type="checkbox" />
      <span>解析完成后自动继续生成</span>
    </label>

    <label class="checkbox-row">
      <input v-model="form.generate_tiles" type="checkbox" />
      <span>预览图完成后继续生成交互瓦片</span>
    </label>

    <button class="primary-button" type="submit" :disabled="submitting">
      {{ submitting ? '提交中...' : '开始生成 Beta 地图' }}
    </button>
  </form>
</template>

<style scoped>
.task-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.task-form label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  color: #666;
}
.task-form textarea,
.task-form input[type="number"] {
  padding: 8px 10px;
  font-size: 13px;
  border: 1px solid #d9d9d9;
  border-radius: 2px;
  background: #fff;
  color: #333;
  font-family: inherit;
}
.task-form textarea:focus,
.task-form input[type="number"]:focus {
  outline: none;
  border-color: #3385ff;
  box-shadow: 0 0 0 2px rgba(51, 133, 255, 0.1);
}
.task-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 6px;
}
.projection-selector {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.selector-label {
  font-size: 13px;
  color: #666;
}
.selector-buttons {
  display: flex;
  gap: 6px;
}
.selector-buttons button {
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
.selector-buttons button.active {
  background: #3385ff;
  color: #fff;
  border-color: #3385ff;
}
.selector-hint {
  font-size: 11px;
  color: #999;
  margin: 0;
}
.checkbox-row {
  flex-direction: row !important;
  align-items: center;
  gap: 6px !important;
}
.checkbox-row input[type="checkbox"] {
  accent-color: #3385ff;
}
.primary-button {
  padding: 9px 16px;
  font-size: 14px;
  font-weight: 500;
  border: none;
  border-radius: 2px;
  background: #3385ff;
  color: #fff;
  cursor: pointer;
  transition: background 0.2s;
}
.primary-button:hover:not(:disabled) {
  background: #5a9fff;
}
.primary-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>
