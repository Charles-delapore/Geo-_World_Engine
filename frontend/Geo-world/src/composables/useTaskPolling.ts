import { onBeforeUnmount, ref, watch, type Ref } from 'vue'

import { getMap } from '../api/maps'
import type { MapTask } from '../api/types'

const MAX_POLL_ATTEMPTS = 200
const POLL_INTERVAL_MS = 3000

export function useTaskPolling(taskId: Ref<string | null>) {
  const task = ref<MapTask | null>(null)
  const isPolling = ref(false)
  const pollError = ref<string | null>(null)
  let timer: number | null = null
  let attemptCount = 0
  let consecutiveErrors = 0
  let inFlight = false

  function stopPolling() {
    if (timer !== null) {
      window.clearInterval(timer)
      timer = null
    }
    isPolling.value = false
    attemptCount = 0
  }

  async function refresh() {
    if (!taskId.value) {
      return
    }
    if (inFlight) {
      return
    }

    attemptCount++
    if (attemptCount > MAX_POLL_ATTEMPTS) {
      pollError.value = 'Polling timed out: max attempts reached'
      stopPolling()
      return
    }

    try {
      inFlight = true
      const next = await getMap(taskId.value)
      consecutiveErrors = 0
      pollError.value = null
      task.value = next
      const waitsForTiles = next.diagnostics?.generateTiles !== false
      if (next.status === 'ready-interactive' || next.status === 'failed' || (next.status === 'ready-image' && !waitsForTiles)) {
        stopPolling()
      }
    } catch (err: unknown) {
      consecutiveErrors++
      const msg = err instanceof Error ? err.message : String(err)
      pollError.value = msg

      if (err && typeof err === 'object' && 'response' in err) {
        const axiosError = err as { response?: { status?: number } }
        if (axiosError.response?.status === 404) {
          pollError.value = `任务不存在或已过期 (${taskId.value?.slice(0, 8)}...)`
          stopPolling()
          return
        }
      }

      if (consecutiveErrors >= 5) {
        pollError.value = `Polling stopped after ${consecutiveErrors} consecutive errors: ${msg}`
        stopPolling()
      }
    } finally {
      inFlight = false
    }
  }

  function startPolling() {
    stopPolling()
    isPolling.value = true
    consecutiveErrors = 0
    pollError.value = null
    void refresh()
    timer = window.setInterval(() => {
      void refresh()
    }, POLL_INTERVAL_MS)
  }

  watch(taskId, (next) => {
    if (!next) {
      stopPolling()
      task.value = null
    }
  })

  onBeforeUnmount(stopPolling)

  return {
    task,
    isPolling,
    pollError,
    refresh,
    startPolling,
    stopPolling,
  }
}
