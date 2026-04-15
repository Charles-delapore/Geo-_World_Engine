<script setup lang="ts">
import { reactive } from 'vue'

export interface TaskFormPayload {
  prompt: string
  width: number
  height: number
  seed?: number
  auto_confirm: boolean
  generate_tiles: boolean
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
})

function onSubmit() {
  emit('submit', {
    prompt: form.prompt,
    width: form.width,
    height: form.height,
    seed: form.seed ? Number(form.seed) : undefined,
    auto_confirm: form.auto_confirm,
    generate_tiles: form.generate_tiles,
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
