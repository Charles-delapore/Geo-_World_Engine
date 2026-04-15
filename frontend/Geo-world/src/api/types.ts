export interface CreateMapRequest {
  prompt: string
  width: number
  height: number
  seed?: number
  auto_confirm: boolean
  generate_tiles: boolean
}

export interface MapTask {
  taskId: string
  status: 'processing' | 'awaiting-confirm' | 'ready-image' | 'ready-interactive' | 'failed'
  currentStage: string | null
  progress: number
  previewUrl: string | null
  manifestUrl: string | null
  errorMsg: string | null
  planSummary: string | null
  createdAt: string
  updatedAt: string
}
