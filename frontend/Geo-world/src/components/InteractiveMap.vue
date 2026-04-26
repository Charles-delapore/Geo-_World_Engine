<script setup lang="ts">
import 'maplibre-gl/dist/maplibre-gl.css'

import maplibregl from 'maplibre-gl'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps<{
  manifestUrl: string | null
  active: boolean
}>()

const host = ref<HTMLDivElement | null>(null)
let map: maplibregl.Map | null = null

async function ensureMap() {
  if (!props.active || !props.manifestUrl || !host.value) {
    return
  }

  const response = await fetch(props.manifestUrl)
  if (!response.ok) {
    throw new Error('加载瓦片 manifest 失败')
  }
  const manifest = await response.json()
  const sourceMaxZoom = manifest.max_zoom ?? 4
  const sourceMinZoom = manifest.min_zoom ?? 0

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
      maxZoom: sourceMaxZoom + 2,
      scrollZoom: true,
      dragPan: true,
      dragRotate: true,
      renderWorldCopies: true,
      pitch: 0,
      bearing: 0,
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right')
    map.scrollZoom.enable()
    map.on('load', () => {
      if (!map) {
        return
      }
      map.addSource('beta-tiles', {
        type: 'raster',
        tiles: [manifest.tile_url_template],
        tileSize: 256,
        minzoom: sourceMinZoom,
        maxzoom: sourceMaxZoom,
      })
      map.addLayer({
        id: 'beta-tiles-layer',
        type: 'raster',
        source: 'beta-tiles',
      })
    })
    return
  }

  if (map.getLayer('beta-tiles-layer')) map.removeLayer('beta-tiles-layer')
  if (map.getSource('beta-tiles')) map.removeSource('beta-tiles')
  map.addSource('beta-tiles', {
    type: 'raster',
    tiles: [manifest.tile_url_template],
    tileSize: 256,
    minzoom: sourceMinZoom,
    maxzoom: sourceMaxZoom,
  })
  map.addLayer({
    id: 'beta-tiles-layer',
    type: 'raster',
    source: 'beta-tiles',
  })
  map.setMaxZoom(sourceMaxZoom + 2)
}

watch(
  () => [props.manifestUrl, props.active] as const,
  () => {
    void ensureMap()
  },
)

onMounted(() => {
  void ensureMap()
})

onBeforeUnmount(() => {
  map?.remove()
  map = null
})
</script>

<template>
  <div class="interactive-shell">
    <div ref="host" class="interactive-map" />
  </div>
</template>
