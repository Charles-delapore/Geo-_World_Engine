import { apiClient } from './client'
import type { CreateMapRequest, MapTask } from './types'

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
