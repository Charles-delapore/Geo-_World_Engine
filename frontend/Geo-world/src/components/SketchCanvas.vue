<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps<{
  active: boolean
  brushMode: 'raise' | 'lower' | 'flatten' | 'roughen'
  brushSize: number
}>()

const emit = defineEmits<{
  (e: 'stroke', geojson: Record<string, unknown>): void
  (e: 'stroke-complete', geojson: Record<string, unknown>): void
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
let isDrawing = false
let currentStroke: Array<[number, number]> = []
let ctx: CanvasRenderingContext2D | null = null

function getCanvasCoords(e: MouseEvent): [number, number] {
  if (!canvas.value) return [0, 0]
  const rect = canvas.value.getBoundingClientRect()
  const x = (e.clientX - rect.left) / rect.width
  const y = (e.clientY - rect.top) / rect.height
  return [x, y]
}

function startStroke(e: MouseEvent) {
  if (!props.active) return
  isDrawing = true
  currentStroke = [getCanvasCoords(e)]
}

function continueStroke(e: MouseEvent) {
  if (!isDrawing || !props.active || !ctx || !canvas.value) return
  const pt = getCanvasCoords(e)
  currentStroke.push(pt)

  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height)
  drawExistingStrokes()

  ctx.beginPath()
  ctx.strokeStyle = props.brushMode === 'raise' ? '#ff9944'
    : props.brushMode === 'lower' ? '#4488ff'
    : props.brushMode === 'flatten' ? '#44cc88'
    : '#cc44ff'
  ctx.lineWidth = props.brushSize * 3
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.globalAlpha = 0.5

  for (let i = 0; i < currentStroke.length; i++) {
    const [x, y] = currentStroke[i]
    const px = x * canvas.value.width
    const py = y * canvas.value.height
    if (i === 0) {
      ctx.moveTo(px, py)
    } else {
      ctx.lineTo(px, py)
    }
  }
  ctx.stroke()
  ctx.globalAlpha = 1.0
}

function endStroke() {
  if (!isDrawing || !props.active) return
  isDrawing = false

  if (currentStroke.length >= 2) {
    const geojson = strokeToGeoJSON(currentStroke)
    emit('stroke-complete', geojson)
  }

  currentStroke = []
  if (ctx && canvas.value) {
    ctx.clearRect(0, 0, canvas.value.width, canvas.value.height)
  }
}

function strokeToGeoJSON(stroke: Array<[number, number]>): Record<string, unknown> {
  const coordinates = stroke.map(([x, y]) => [x * 360 - 180, 90 - y * 180])
  return {
    type: 'Feature',
    properties: {
      brushMode: props.brushMode,
      brushSize: props.brushSize,
    },
    geometry: {
      type: 'LineString',
      coordinates,
    },
  }
}

function drawExistingStrokes() {
}

watch(() => props.active, (active) => {
  if (!active && ctx && canvas.value) {
    ctx.clearRect(0, 0, canvas.value.width, canvas.value.height)
  }
})

onMounted(() => {
  if (canvas.value) {
    ctx = canvas.value.getContext('2d')
    const resize = () => {
      if (!canvas.value) return
      const parent = canvas.value.parentElement
      if (parent) {
        canvas.value.width = parent.clientWidth
        canvas.value.height = parent.clientHeight
      }
    }
    resize()
    window.addEventListener('resize', resize)
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', () => {})
})
</script>

<template>
  <canvas
    v-show="active"
    ref="canvas"
    class="sketch-canvas"
    @mousedown="startStroke"
    @mousemove="continueStroke"
    @mouseup="endStroke"
    @mouseleave="endStroke"
  />
</template>

<style scoped>
.sketch-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  cursor: crosshair;
  z-index: 10;
  pointer-events: auto;
}
</style>
