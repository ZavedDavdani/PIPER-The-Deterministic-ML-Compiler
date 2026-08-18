import { useRef, useState } from 'react'
import { UploadCloud, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { ApiError, uploadDataset } from '@/lib/api'
import type { DatasetUploadResponse } from '@/lib/types'
import { cn } from '@/lib/utils'

interface DatasetUploadProps {
  onUploaded: (dataset: DatasetUploadResponse) => void
}

/**
 * Mirrors the backend's FORMAT_EXTENSIONS allowlist
 * (app/schemas/ingestion.py). Client-side checking is a UX
 * convenience only — the backend independently re-validates and is
 * the authority, exactly as it is for every other input.
 */
const ACCEPTED_EXTENSIONS = [
  '.csv',
  '.tsv',
  '.tab',
  '.xlsx',
  '.xlsm',
  '.xls',
  '.json',
  '.ipynb',
  '.parquet',
  '.pq',
]

const FORMAT_LABELS: Record<string, string> = {
  csv: 'CSV',
  tsv: 'TSV',
  excel: 'Excel',
  json: 'JSON',
  ipynb: 'Jupyter notebook',
  parquet: 'Parquet',
}

export function DatasetUpload({ onUploaded }: DatasetUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpload, setLastUpload] = useState<DatasetUploadResponse | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function handleFile(file: File) {
    setError(null)
    const name = file.name.toLowerCase()
    if (!ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext))) {
      setError(`Unsupported file type. Accepted formats: ${ACCEPTED_EXTENSIONS.join(', ')}.`)
      return
    }
    setIsUploading(true)
    try {
      const dataset = await uploadDataset(file)
      setLastUpload(dataset)
      onUploaded(dataset)
    } catch (e) {
      setLastUpload(null)
      setError(e instanceof ApiError ? e.message : 'Upload failed — is the PIPER backend running?')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          const file = e.dataTransfer.files[0]
          if (file) void handleFile(file)
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload a dataset"
        className={cn(
          'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors',
          isDragging ? 'border-primary bg-accent' : 'border-border hover:bg-muted/50',
        )}
      >
        {isUploading ? (
          <Loader2 className="text-muted-foreground size-8 animate-spin" />
        ) : (
          <UploadCloud className="text-muted-foreground size-8" />
        )}
        <p className="text-sm font-medium">
          {isUploading ? 'Uploading…' : 'Drop a dataset here, or click to browse'}
        </p>
        <p className="text-muted-foreground text-xs">
          CSV, TSV, Excel, JSON, Jupyter notebook, or Parquet — one row per observation.
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          aria-label="Choose dataset file"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void handleFile(file)
            e.target.value = ''
          }}
        />
      </div>
      {error && (
        <Alert variant="destructive">
          <AlertTitle>Upload failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {lastUpload && !error && (
        <div className="border-border bg-muted/40 flex flex-col gap-1 rounded-md border p-3 text-xs">
          <p className="text-foreground font-medium">
            Detected {FORMAT_LABELS[lastUpload.detected_format ?? ''] ?? lastUpload.detected_format ?? 'file'} —{' '}
            {lastUpload.rows.toLocaleString()} rows × {lastUpload.column_count ?? lastUpload.columns.length} columns
          </p>
          {lastUpload.sheet_name && (
            <p className="text-muted-foreground">
              Worksheet: <span className="font-mono">{lastUpload.sheet_name}</span>
              {lastUpload.available_sheets && lastUpload.available_sheets.length > 1 && (
                <> of {lastUpload.available_sheets.length}</>
              )}
            </p>
          )}
          {lastUpload.notes?.map((note) => (
            <p key={note} className="text-muted-foreground">
              {note}
            </p>
          ))}
        </div>
      )}
      <Button type="button" variant="outline" size="sm" onClick={() => inputRef.current?.click()} disabled={isUploading}>
        Choose file
      </Button>
    </div>
  )
}
