<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps<{
  manifestUrl: string | null
  previewUrl: string | null
  active: boolean
  editTimestamp: number
}>()

const emit = defineEmits<{
  (e: 'center-change', center: [number, number]): void
  (e: 'zoom-change', zoom: number): void
  (e: 'zoom-range', min: number, max: number): void
}>()

const host = ref<HTMLDivElement | null>(null)
const loading = ref(true)
const errorMsg = ref<string | null>(null)
let viewer: any = null
let currentLayerKey: string | null = null
let moveHandler: any = null
let CesiumApi: any = null
let cameraGuardTimer: number | null = null
let dragCleanup: (() => void) | null = null
let orbitHeading = 0
let orbitPitch = -1.25
let targetCameraRange = 22_000_000

const MIN_CAMERA_RANGE = 13_000_000
const MAX_CAMERA_RANGE = 38_000_000

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
  return template
}

function setupMouseHandler(Cesium: any) {
  if (!viewer) return

  if (moveHandler) moveHandler.destroy?.()
  moveHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas)
  moveHandler.setInputAction((movement: any) => {
    const cartesian = viewer.camera.pickEllipsoid(movement.endPosition, viewer.scene.globe.ellipsoid)
    if (cartesian) {
      const cartographic = Cesium.Cartographic.fromCartesian(cartesian)
      const lng = Cesium.Math.toDegrees(cartographic.longitude)
      const lat = Cesium.Math.toDegrees(cartographic.latitude)
      emit('center-change', [lng, lat])
    }
  }, Cesium.ScreenSpaceEventType.MOUSE_MOVE)
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function applyOrbitCamera(Cesium: any) {
  if (!viewer) return
  try {
    targetCameraRange = clamp(targetCameraRange, MIN_CAMERA_RANGE, MAX_CAMERA_RANGE)
    const target = Cesium.Cartesian3.ZERO
    const offset = new Cesium.HeadingPitchRange(orbitHeading, orbitPitch, targetCameraRange)
    viewer.camera.lookAt(target, offset)
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY)
    viewer.scene.requestRender?.()
    emit('zoom-change', estimateZoomFromRange(targetCameraRange))
  } catch {
    // Ignore camera control failures to avoid breaking the view.
  }
}

function estimateZoomFromRange(range: number) {
  const normalized = Math.log2(MAX_CAMERA_RANGE / Math.max(range, 1))
  return clamp(Math.round(normalized * 1.6), 0, 8)
}

function lockCameraController() {
  if (!viewer) return
  const controller = viewer.scene.screenSpaceCameraController
  controller.enableInputs = false
  controller.enableZoom = false
  controller.enableRotate = false
  controller.enableTranslate = false
  controller.enableTilt = false
  controller.enableLook = false
  controller.enableCollisionDetection = false
  controller.inertiaZoom = 0
  controller.inertiaSpin = 0
  controller.inertiaTranslate = 0
  controller.minimumZoomDistance = MIN_CAMERA_RANGE
  controller.maximumZoomDistance = MAX_CAMERA_RANGE
}

function startCameraGuard(Cesium: any) {
  if (cameraGuardTimer !== null) {
    window.clearInterval(cameraGuardTimer)
  }
  cameraGuardTimer = window.setInterval(() => {
    lockCameraController()
    applyOrbitCamera(Cesium)
  }, 250)
}

function stopCameraGuard() {
  if (cameraGuardTimer !== null) {
    window.clearInterval(cameraGuardTimer)
    cameraGuardTimer = null
  }
}

function setupDragHandler(Cesium: any) {
  if (!viewer) return
  dragCleanup?.()
  const canvas = viewer.scene.canvas
  let dragging = false
  let lastX = 0
  let lastY = 0

  const onPointerDown = (event: PointerEvent) => {
    dragging = true
    lastX = event.clientX
    lastY = event.clientY
    canvas.setPointerCapture?.(event.pointerId)
  }
  const onPointerMove = (event: PointerEvent) => {
    if (!dragging) return
    const dx = event.clientX - lastX
    const dy = event.clientY - lastY
    lastX = event.clientX
    lastY = event.clientY
    orbitHeading -= dx * 0.004
    orbitPitch = clamp(orbitPitch + dy * 0.002, -1.48, -0.35)
    applyOrbitCamera(Cesium)
  }
  const onPointerUp = (event: PointerEvent) => {
    dragging = false
    canvas.releasePointerCapture?.(event.pointerId)
  }

  canvas.addEventListener('pointerdown', onPointerDown)
  canvas.addEventListener('pointermove', onPointerMove)
  canvas.addEventListener('pointerup', onPointerUp)
  canvas.addEventListener('pointercancel', onPointerUp)
  dragCleanup = () => {
    canvas.removeEventListener('pointerdown', onPointerDown)
    canvas.removeEventListener('pointermove', onPointerMove)
    canvas.removeEventListener('pointerup', onPointerUp)
    canvas.removeEventListener('pointercancel', onPointerUp)
    dragCleanup = null
  }
}

async function ensureViewer() {
  if (!props.active || !host.value || (!props.manifestUrl && !props.previewUrl)) return

  const layerKey = `${props.previewUrl ?? ''}|${props.manifestUrl ?? ''}|${String(props.editTimestamp ?? 0)}`
  if (currentLayerKey === layerKey && viewer) return
  currentLayerKey = layerKey

  try {
    if (viewer) {
      try {
        const Cesium = await import('cesium')
        CesiumApi = Cesium
        const provider = await createProvider(Cesium)
        viewer.imageryLayers.removeAll()
        viewer.imageryLayers.addImageryProvider(provider)
        try {
          const surfaceTiles = (viewer.scene.globe as any)?._surface?._tiles
          if (surfaceTiles) {
            for (const tile of surfaceTiles) {
              tile?.freeResources?.()
            }
          }
        } catch {}
        try {
          const surface = (viewer.scene.globe as any)?._surface
          if (surface?._tileProvider?.invalidateAllTiles) {
            surface._tileProvider.invalidateAllTiles()
          }
        } catch {}
        viewer.scene.requestRender()
        lockCameraController()
        applyOrbitCamera(Cesium)
        loading.value = false
        errorMsg.value = null
      } catch (e) {
        console.error('Failed to update imagery:', e)
        errorMsg.value = '瓦片图层更新失败'
        destroyViewer()
        await createNewViewer()
      }
      return
    }

    await createNewViewer()
  } catch (e) {
    console.error('Failed to load manifest:', e)
    errorMsg.value = '地图清单加载失败'
  }
}

async function createProvider(Cesium: any) {
  if (!props.manifestUrl && props.previewUrl) {
    return Cesium.SingleTileImageryProvider.fromUrl(resolveUrl(props.previewUrl), {
      rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
    })
  }

  if (!props.manifestUrl) {
    throw new Error('Manifest URL is required when preview texture is unavailable')
  }

  const response = await fetch(props.manifestUrl, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`无法加载清单 (${response.status})`)
  }
  const manifest = await response.json()
  if (manifest.tile_url_template) {
    manifest.tile_url_template = resolveUrl(manifest.tile_url_template)
  }

  const isGeographic = manifest.tiling_scheme === 'geographic'
  const tilingScheme = isGeographic
    ? new Cesium.GeographicTilingScheme({
        numberOfLevelZeroTilesX: manifest.level_zero_tiles_x ?? 1,
        numberOfLevelZeroTilesY: manifest.level_zero_tiles_y ?? 1,
        rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
      })
    : new Cesium.WebMercatorTilingScheme()

  return new Cesium.UrlTemplateImageryProvider({
    url: manifest.tile_url_template as string,
    minimumLevel: manifest.min_zoom ?? 0,
    maximumLevel: manifest.max_zoom ?? 4,
    tileWidth: 256,
    tileHeight: 256,
    tilingScheme,
  })
}

async function createNewViewer() {
  if (!host.value) return
  const Cesium = await import('cesium')
  CesiumApi = Cesium
  await import('cesium/Build/Cesium/Widgets/widgets.css')

  Cesium.Ion.defaultAccessToken = ''

  try {
    const provider = await createProvider(Cesium)

    viewer = new Cesium.Viewer(host.value, {
      baseLayer: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      terrain: undefined,
      skyBox: false,
      skyAtmosphere: new Cesium.SkyAtmosphere(),
      requestRenderMode: true,
    })

    viewer.imageryLayers.addImageryProvider(provider)

    viewer.scene.backgroundColor = Cesium.Color.BLACK
    viewer.scene.globe.enableLighting = false
    viewer.scene.globe.showGroundAtmosphere = false
    viewer.scene.fog.enabled = false
    lockCameraController()

    applyOrbitCamera(Cesium)

    emit('zoom-range', 0, 8)

    setupMouseHandler(Cesium)
    setupDragHandler(Cesium)
    startCameraGuard(Cesium)

    let lastZoomEmit = 0
    viewer.camera.changed.addEventListener(() => {
      const now = Date.now()
      if (now - lastZoomEmit < 500) return
      lastZoomEmit = now
      try {
        emit('zoom-change', estimateZoomFromRange(targetCameraRange))
      } catch {}
    })
    viewer.camera.percentageChanged = 0.01

    loading.value = false
    errorMsg.value = null
  } catch (e) {
    console.error('Failed to create Cesium viewer:', e)
    errorMsg.value = `星球视图创建失败: ${e instanceof Error ? e.message : String(e)}`
    loading.value = false
  }
}

function destroyViewer() {
  stopCameraGuard()
  dragCleanup?.()
  if (moveHandler) {
    moveHandler.destroy?.()
    moveHandler = null
  }
  if (viewer) {
    viewer.destroy()
    viewer = null
  }
  CesiumApi = null
  currentLayerKey = null
}

function zoomIn() {
  if (!viewer || !CesiumApi) return
  targetCameraRange = clamp(targetCameraRange * 0.82, MIN_CAMERA_RANGE, MAX_CAMERA_RANGE)
  applyOrbitCamera(CesiumApi)
}

function zoomOut() {
  if (!viewer || !CesiumApi) return
  targetCameraRange = clamp(targetCameraRange * 1.22, MIN_CAMERA_RANGE, MAX_CAMERA_RANGE)
  applyOrbitCamera(CesiumApi)
}

defineExpose({ zoomIn, zoomOut, refreshTiles })

function refreshTiles() {
  invalidateAllGlobeTiles()
  if (viewer) {
    viewer.scene.requestRender()
  }
}

function invalidateAllGlobeTiles() {
  try {
    const surface = (viewer?.scene?.globe as any)?._surface
    if (surface?._tileProvider?.invalidateAllTiles) {
      surface._tileProvider.invalidateAllTiles()
    }
  } catch {}
  try {
    const tiles = (viewer?.scene?.globe as any)?._surface?._tiles
    if (tiles) {
      for (const tile of tiles) {
        tile?.freeResources?.()
      }
    }
  } catch {}
}

watch(
  () => [props.manifestUrl, props.previewUrl, props.active, props.editTimestamp] as const,
  () => {
    void ensureViewer()
  },
)

onMounted(() => {
  void ensureViewer()
})

onBeforeUnmount(() => {
  destroyViewer()
})
</script>

<template>
  <div class="cesium-shell">
    <div ref="host" class="cesium-container" />
    <div v-if="loading" class="cesium-loading">
      <div class="spinner" />
      <span>加载星球视图…</span>
    </div>
    <div v-if="errorMsg" class="cesium-error">
      <span>{{ errorMsg }}</span>
    </div>
  </div>
</template>

<style scoped>
.cesium-shell {
  width: 100%;
  height: 100%;
  position: relative;
}
.cesium-container {
  width: 100%;
  height: 100%;
  min-height: 520px;
  position: absolute;
  inset: 0;
  overflow: hidden;
  contain: layout size paint;
}
.cesium-container :deep(.cesium-viewer),
.cesium-container :deep(.cesium-viewer-cesiumWidgetContainer),
.cesium-container :deep(.cesium-widget),
.cesium-container :deep(.cesium-widget canvas) {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
}
.cesium-container :deep(.cesium-viewer-bottom) {
  display: none !important;
}
.cesium-container :deep(.cesium-viewer-toolbar) {
  display: none !important;
}
.cesium-loading {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  color: #999;
  font-size: 13px;
  pointer-events: none;
  z-index: 10;
}
.cesium-error {
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
  padding: 6px 14px;
  background: rgba(255, 77, 79, 0.9);
  color: #fff;
  font-size: 12px;
  border-radius: 2px;
  z-index: 10;
  pointer-events: none;
  max-width: 80%;
  text-align: center;
}
.spinner {
  width: 28px;
  height: 28px;
  border: 3px solid #e8e8e8;
  border-top-color: #3385ff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
