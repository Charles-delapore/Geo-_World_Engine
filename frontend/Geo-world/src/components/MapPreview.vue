<script setup lang="ts">
defineProps<{
  previewUrl: string | null
  visible: boolean
}>()
</script>

<template>
  <div v-show="visible" class="preview-shell">
    <img
      v-if="previewUrl"
      :src="previewUrl"
      class="preview-image"
      alt="World preview"
    />
  </div>
</template>
