import type {
  ApiErrorBody,
  CreateRunRequest,
  CreateRunResponse,
  DatasetListResponse,
  DatasetProfile,
  DatasetUploadResponse,
  DecisionTrace,
  EvidenceExport,
  HumanInterventionPackage,
  OllamaStatusResponse,
  PiperVerdict,
  ReplayResponse,
  RunListResponse,
  RunResultResponse,
  RunStatusResponse,
  ArtifactStatusResponse,
  ArtifactFileListResponse,
} from './types'

/**
 * Base URL of the real PIPER FastAPI backend. Configurable via
 * VITE_API_BASE_URL (set at build/dev time) so the same build can
 * target a local dev backend, a Docker-composed one, or anything
 * else — never hardcoded to a single environment. Defaults to the
 * conventional local dev backend port (uvicorn's default is 8000).
 */
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as ApiErrorBody
      if (body?.detail) detail = body.detail
    } catch {
      // response body wasn't JSON — fall back to statusText
    }
    throw new ApiError(response.status, detail)
  }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

// --- Datasets --------------------------------------------------------------

export function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  return request<DatasetUploadResponse>('/datasets', { method: 'POST', body: formData })
}

export function listDatasets(): Promise<DatasetListResponse> {
  return request<DatasetListResponse>('/datasets')
}

export function getDataset(datasetId: string): Promise<DatasetProfile> {
  return request<DatasetProfile>(`/datasets/${encodeURIComponent(datasetId)}`)
}

// --- Runs --------------------------------------------------------------------

export function createRun(body: CreateRunRequest): Promise<CreateRunResponse> {
  return request<CreateRunResponse>('/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function getRunStatus(runId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/runs/${encodeURIComponent(runId)}`)
}

/** Throws ApiError(409) if the run hasn't reached a terminal status yet — callers should treat that as "not ready", not a real error. */
export function getRunResult(runId: string): Promise<RunResultResponse> {
  return request<RunResultResponse>(`/runs/${encodeURIComponent(runId)}/result`)
}

export function runEventsUrl(runId: string): string {
  return `${API_BASE_URL}/runs/${encodeURIComponent(runId)}/events`
}

export function getDecisionTrace(runId: string): Promise<DecisionTrace> {
  return request<DecisionTrace>(`/runs/${encodeURIComponent(runId)}/decision-trace`)
}

export function getVerdict(runId: string): Promise<PiperVerdict> {
  return request<PiperVerdict>(`/runs/${encodeURIComponent(runId)}/verdict`)
}

export function getIntervention(runId: string): Promise<HumanInterventionPackage> {
  return request<HumanInterventionPackage>(`/runs/${encodeURIComponent(runId)}/intervention`)
}

export function getEvidence(runId: string): Promise<EvidenceExport> {
  return request<EvidenceExport>(`/runs/${encodeURIComponent(runId)}/evidence`)
}

export function listRuns(): Promise<RunListResponse> {
  return request<RunListResponse>('/runs')
}

export function replayRun(runId: string): Promise<ReplayResponse> {
  return request<ReplayResponse>(`/runs/${encodeURIComponent(runId)}/replay`)
}

export function getOllamaStatus(): Promise<OllamaStatusResponse> {
  return request<OllamaStatusResponse>('/settings/ollama')
}

export function updateOllamaConfig(body: { model?: string; host?: string }): Promise<OllamaStatusResponse> {
  return request<OllamaStatusResponse>('/settings/ollama', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function getArtifactStatus(runId: string): Promise<ArtifactStatusResponse> {
  return request<ArtifactStatusResponse>(`/runs/${encodeURIComponent(runId)}/artifacts`)
}

export function generateArtifacts(runId: string): Promise<ArtifactStatusResponse> {
  return request<ArtifactStatusResponse>(`/runs/${encodeURIComponent(runId)}/artifacts`, {
    method: 'POST',
  })
}

export function listArtifactFiles(runId: string): Promise<ArtifactFileListResponse> {
  return request<ArtifactFileListResponse>(`/runs/${encodeURIComponent(runId)}/artifacts/files`)
}

export function artifactFileUrl(runId: string, filename: string): string {
  return `${API_BASE_URL}/runs/${encodeURIComponent(runId)}/artifacts/files/${encodeURIComponent(filename)}`
}
