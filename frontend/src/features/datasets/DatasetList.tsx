import { useEffect, useState } from 'react'
import { Database, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { ApiError, listDatasets } from '@/lib/api'
import { cn } from '@/lib/utils'

interface DatasetListProps {
  selectedId: string | null
  onSelect: (datasetId: string) => void
  refreshToken: number
}

export function DatasetList({ selectedId, onSelect, refreshToken }: DatasetListProps) {
  const [datasetIds, setDatasetIds] = useState<string[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    setError(null)
    try {
      const response = await listDatasets()
      setDatasetIds(response.dataset_ids)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not reach the PIPER backend.')
      setDatasetIds([])
    }
  }

  useEffect(() => {
    void load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshToken])

  if (error) {
    return <p className="text-destructive text-sm">{error}</p>
  }

  if (datasetIds === null) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground text-xs">{datasetIds.length} dataset{datasetIds.length === 1 ? '' : 's'}</span>
        <Button type="button" variant="ghost" size="sm" onClick={() => void load()} aria-label="Refresh dataset list">
          <RefreshCw className="size-3.5" />
        </Button>
      </div>
      {datasetIds.length === 0 && (
        <p className="text-muted-foreground text-sm">No datasets uploaded yet.</p>
      )}
      <ul className="flex flex-col gap-1">
        {datasetIds.map((id) => (
          <li key={id}>
            <button
              type="button"
              onClick={() => onSelect(id)}
              className={cn(
                'flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-sm transition-colors',
                id === selectedId ? 'border-primary bg-accent text-accent-foreground' : 'border-border hover:bg-muted/50',
              )}
            >
              <Database className="size-4 shrink-0" />
              <span className="truncate font-mono text-xs">{id}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
