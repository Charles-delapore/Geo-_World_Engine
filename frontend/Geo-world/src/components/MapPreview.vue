<script setup lang="ts">
defineProps<{
  previewUrl: string | null
  visible: boolean
}>()
</script>

<template>
  <div class="preview-shell">
    <img
      v-if="previewUrl && visible"
      class="preview-image"
      :src="previewUrl"
      alt="World preview"
    />
  </div>
</template>
