import { onBeforeUnmount, ref, watch, type Ref } from 'vue'

import { getMap } from '../api/maps'
import type { MapTask } from '../api/types'

export function useTaskPolling(taskId: Ref<string | null>) {
  const task = ref<MapTask | null>(null)
  const isPolling = ref(false)
  let timer: number | null = null

  function stopPolling() {
    if (timer !== null) {
      window.clearInterval(timer)
      timer = null
    }
    isPolling.value = false
  }

  async function refresh() {
    if (!taskId.value) {
      return
    }
    const next = await getMap(taskId.value)
    task.value = next
    if (next.status !== 'processing') {
      stopPolling()
    }
  }

  function startPolling() {
    stopPolling()
    isPolling.value = true
    void refresh()
    timer = window.setInterval(() => {
      void refresh()
    }, 2500)
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
    refresh,
    startPolling,
    stopPolling,
  }
}
