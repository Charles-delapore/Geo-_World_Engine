export interface CreateMapRequest {
  prompt: string
  width: number
  height: number
  seed?: number
  auto_confirm: boolean
  generate_tiles: boolean
  projection: 'flat' | 'planet'
}

export interface MapDiagnostics {
  seed?: number
  width: number
  height: number
  generationBackend: string
  projection?: 'flat' | 'planet'
  generateTiles?: boolean
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

export interface EditRequest {
  instructions?: Array<Record<string, unknown>>
  text_instruction?: string
  sketch_geojson?: Record<string, unknown>
}

export interface EditResponse {
  taskId: string
  versionNum: number
  editSummary: string
  metricReport: Record<string, unknown> | null
}

export interface VersionInfo {
  versionNum: number
  editSummary: string | null
  editType: string | null
  parentVersion: number | null
  createdAt: string
}

export interface TerrainFeature {
  type: string
  area?: number
  count?: number
  bounds?: { min_y: number; min_x: number; max_y: number; max_x: number }
}
