import { apiClient } from './client'
import type { CreateMapRequest, MapTask, EditRequest, EditResponse, VersionInfo, TerrainFeature } from './types'

export async function createMap(payload: CreateMapRequest) {
  const response = await apiClient.post<MapTask>('/maps', payload)
  return response.data
}

export async function getMap(taskId: string) {
  const response = await apiClient.get<MapTask>(`/maps/${taskId}`)
  return response.data
}

export async function confirmMap(taskId: string) {
  await apiClient.post(`/maps/${taskId}/confirm`)
}

export async function editMap(taskId: string, payload: EditRequest) {
  const response = await apiClient.put<EditResponse>(`/maps/${taskId}/edit`, payload)
  return response.data
}

export async function applySketch(taskId: string, geojson: Record<string, unknown>, instructionType: string = 'elevation_offset', brushSize: number = 5) {
  const response = await apiClient.post<EditResponse>(`/maps/${taskId}/sketch`, { geojson, instruction_type: instructionType, brush_size: brushSize })
  return response.data
}

export async function getFeatures(taskId: string, types: string = 'ridge,river,lake') {
  const response = await apiClient.get<{ taskId: string; features: TerrainFeature[] }>(`/maps/${taskId}/features`, { params: { types } })
  return response.data
}

export async function getHistory(taskId: string) {
  const response = await apiClient.get<VersionInfo[]>(`/maps/${taskId}/history`)
  return response.data
}

export async function revertToVersion(taskId: string, versionNum: number) {
  const response = await apiClient.post<EditResponse>(`/maps/${taskId}/revert`, { version_num: versionNum })
  return response.data
}
