<script setup lang="ts">
import 'maplibre-gl/dist/maplibre-gl.css'

import maplibregl from 'maplibre-gl'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps<{
  manifestUrl: string | null
  previewUrl: string | null
  active: boolean
  projection: 'flat' | 'planet'
}>()

const emit = defineEmits<{
  (e: 'zoom-change', zoom: number): void
  (e: 'center-change', center: [number, number]): void
  (e: 'zoom-range', min: number, max: number): void
}>()

const host = ref<HTMLDivElement | null>(null)
let map: maplibregl.Map | null = null
let isReady = false
let zooming = false
let currentMinZoom = 0
let currentMaxZoom = 6

function resolveUrl(url: string): string {
  const template = url
    .replaceAll('%7Bz%7D', '{z}')
    .replaceAll('%7Bx%7D', '{x}')
    .replaceAll('%7By%7D', '{y}')
  if (url.startsWith('/')) {
    return `${window.location.origin}${template}`
  }
  if (url.startsWith('http://') || url.startsWith('https://')) {
    const apiIndex = url.indexOf('/api/')
    if (apiIndex >= 0) {
      return `${window.location.origin}${template.slice(apiIndex)}`
    }
  }
  if (url.startsWith('http://backend-api') || url.startsWith('https://backend-api')) {
    const path = new URL(url).pathname
    return `${window.location.origin}${path}`
  }
  return template
}

function syncState() {
  if (!map) return
  const zoom = map.getZoom()
  const center = map.getCenter()
  if (!isFinite(zoom) || zoom < -10 || zoom > 30) return
  if (!isFinite(center.lng) || !isFinite(center.lat)) return
  emit('zoom-change', Number(Math.min(Math.max(zoom, currentMinZoom), currentMaxZoom).toFixed(2)))
  emit('center-change', [center.lng, center.lat])
}

async function waitForLayout() {
  await nextTick()
  if (!host.value) return
  if (host.value.clientWidth > 0 && host.value.clientHeight > 0) return
  await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()))
}

function manifestBounds(manifest: Record<string, any>): [number, number, number, number] | undefined {
  if (!Array.isArray(manifest.bounds) || manifest.bounds.length !== 4) return undefined
  const bounds = manifest.bounds.map(Number)
  if (bounds.some((value) => !isFinite(value))) return undefined
  return [bounds[0], bounds[1], bounds[2], bounds[3]]
}

function rasterSourceFor(manifest: Record<string, any>, sourceMinZoom: number, sourceMaxZoom: number): maplibregl.RasterSourceSpecification {
  return {
    type: 'raster',
    tiles: [manifest.tile_url_template],
    tileSize: 256,
    minzoom: sourceMinZoom,
    maxzoom: sourceMaxZoom,
    bounds: manifestBounds(manifest),
  }
}

function fitManifestBounds(manifest: Record<string, any>) {
  if (!map) return
  const bounds = manifestBounds(manifest)
  if (!bounds) return
  const [west, south, east, north] = bounds
  map.fitBounds([[west, south], [east, north]], { duration: 0, padding: 0 })
}

function previewCoordinates(manifest: Record<string, any>): [[number, number], [number, number], [number, number], [number, number]] {
  const bounds = manifestBounds(manifest) ?? [-180, -85, 180, 85]
  const [west, south, east, north] = bounds
  return [[west, north], [east, north], [east, south], [west, south]]
}

function addPreviewFallback(manifest: Record<string, any>) {
  if (!map || !props.previewUrl) return
  if (map.getLayer('preview-fallback-layer')) map.removeLayer('preview-fallback-layer')
  if (map.getSource('preview-fallback')) map.removeSource('preview-fallback')
  map.addSource('preview-fallback', {
    type: 'image',
    url: resolveUrl(props.previewUrl),
    coordinates: previewCoordinates(manifest),
  })
  map.addLayer({
    id: 'preview-fallback-layer',
    type: 'raster',
    source: 'preview-fallback',
    paint: { 'raster-opacity': 1.0 },
  })
}

async function ensureMap() {
  if (!props.active || !host.value || (!props.manifestUrl && !props.previewUrl)) {
    return
  }

  try {
    let manifest: Record<string, any> = {
      bounds: [-180, -85, 180, 85],
      center: [0, 0],
      min_zoom: 0,
      max_zoom: 6,
      projection: props.projection,
      tiling_scheme: props.projection === 'planet' ? 'geographic' : 'web_mercator',
    }
    if (props.manifestUrl) {
      const manifestFetchUrl = resolveUrl(props.manifestUrl)
      const response = await fetch(manifestFetchUrl)
      if (!response.ok) {
        console.error('Failed to load tile manifest:', response.status)
      } else {
        manifest = await response.json()
      }
    }

    if (manifest.tile_url_template) {
      manifest.tile_url_template = resolveUrl(manifest.tile_url_template)
    }

    const sourceMaxZoom = manifest.max_zoom ?? 4
    const sourceMinZoom = manifest.min_zoom ?? 0
    currentMinZoom = sourceMinZoom
    currentMaxZoom = sourceMaxZoom + 2
    const isGeographic = manifest.tiling_scheme === 'geographic' || manifest.projection === 'planet' || props.projection === 'planet'
    const maxLat = isGeographic ? 90 : 85
    const minLat = isGeographic ? -90 : -85
    emit('zoom-range', currentMinZoom, currentMaxZoom)

    await waitForLayout()

    if (!map) {
      map = new maplibregl.Map({
        container: host.value,
        style: {
          version: 8,
          sources: {},
          layers: [],
        },
        center: manifest.center ?? [0, 0],
        zoom: sourceMinZoom,
        minZoom: sourceMinZoom,
        maxZoom: currentMaxZoom,
        scrollZoom: true,
        dragPan: true,
        dragRotate: false,
        renderWorldCopies: props.projection === 'planet',
        pitch: 0,
        bearing: 0,
        interactive: true,
        attributionControl: false,
      })
      map.scrollZoom.enable()
      map.doubleClickZoom.enable()
      map.touchZoomRotate.enable()

      map.once('load', () => {
        isReady = true
        map?.resize()
        map?.setMaxBounds([[-180, minLat], [180, maxLat]])
        if (!map) return
        addPreviewFallback(manifest)
        if (manifest.tile_url_template) {
          map.addSource('beta-tiles', rasterSourceFor(manifest, sourceMinZoom, sourceMaxZoom))
          map.addLayer({
            id: 'beta-tiles-layer',
            type: 'raster',
            source: 'beta-tiles',
            paint: { 'raster-opacity': 1.0 },
          })
        }
        fitManifestBounds(manifest)
        syncState()
      })

      map.on('zoom', syncState)
      map.on('move', syncState)
      return
    }

    if (!isReady || !map) return

    if (map.getLayer('beta-tiles-layer')) map.removeLayer('beta-tiles-layer')
    if (map.getSource('beta-tiles')) map.removeSource('beta-tiles')
    addPreviewFallback(manifest)
    if (manifest.tile_url_template) {
      map.addSource('beta-tiles', rasterSourceFor(manifest, sourceMinZoom, sourceMaxZoom))
      map.addLayer({
        id: 'beta-tiles-layer',
        type: 'raster',
        source: 'beta-tiles',
        paint: { 'raster-opacity': 1.0 },
      })
    }
    map.setMaxZoom(currentMaxZoom)

    map.setRenderWorldCopies(props.projection === 'planet')
    map.setMaxBounds([[-180, minLat], [180, maxLat]])
    fitManifestBounds(manifest)

    setTimeout(() => {
      map?.resize()
    }, 100)
  } catch (e) {
    console.error('InteractiveMap ensureMap failed:', e)
  }
}

function zoomIn() {
  if (!isReady || !map || zooming) return
  zooming = true
  map.zoomIn({ duration: 200 })
  setTimeout(() => { zooming = false }, 250)
}

function zoomOut() {
  if (!isReady || !map || zooming) return
  zooming = true
  map.zoomOut({ duration: 200 })
  setTimeout(() => { zooming = false }, 250)
}

defineExpose({ zoomIn, zoomOut })

watch(
  () => [props.manifestUrl, props.previewUrl, props.active, props.projection] as const,
  () => {
    void ensureMap()
  },
)

watch(
  () => props.active,
  (active) => {
    if (active && map) {
      setTimeout(() => {
        map?.resize()
      }, 50)
    }
  },
)

onMounted(() => {
  void ensureMap()
})

onBeforeUnmount(() => {
  isReady = false
  zooming = false
  map?.remove()
  map = null
})
</script>

<template>
  <div class="interactive-shell">
    <img
      v-if="previewUrl"
      class="interactive-fallback"
      :src="resolveUrl(previewUrl)"
      alt=""
    />
    <div ref="host" class="interactive-map" />
  </div>
</template>

<style scoped>
.interactive-shell {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  min-height: 520px;
  pointer-events: auto;
}

.interactive-fallback {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  pointer-events: none;
  z-index: 0;
}

.interactive-map {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  min-height: 520px;
  z-index: 2;
  background: transparent;
  pointer-events: auto;
}

.interactive-map :deep(.maplibregl-canvas),
.interactive-map :deep(.maplibregl-canvas-container) {
  pointer-events: auto;
}
</style>
