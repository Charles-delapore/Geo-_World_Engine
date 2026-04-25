export interface CreateMapRequest {
  prompt: string
  width: number
  height: number
  seed?: number
  auto_confirm: boolean
  generate_tiles: boolean
}

export interface MapDiagnostics {
  seed?: number
  width: number
  height: number
  generationBackend: string
  topologyIntent: string | null
  topologyModifiers: Record<string, string>
  layoutTemplate: string
  seaStyle: string
  landRatio: number
  ruggedness: number
  coastComplexity: number
  islandFactor: number
  moisture: number
  temperatureBias: number
  windDirection: string
  continentCount: number
  mountainCount: number
  seaZoneCount: number
  waterBodyCount: number
  regionalRelationCount: number
  ragEnabled: boolean
  ragExamples: number
  ragTopSimilarity: number | null
  ragFallbackReason: string | null
  metricReport: Record<string, unknown> | null
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
  diagnostics: MapDiagnostics
  createdAt: string
  updatedAt: string
}
