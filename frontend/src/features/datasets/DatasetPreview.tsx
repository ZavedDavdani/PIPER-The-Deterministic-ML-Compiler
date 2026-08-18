import { useEffect, useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ApiError, getDataset } from '@/lib/api'
import type { DatasetProfile } from '@/lib/types'

interface DatasetPreviewProps {
  datasetId: string
  onProfileLoaded?: (profile: DatasetProfile) => void
}

export function DatasetPreview({ datasetId, onProfileLoaded }: DatasetPreviewProps) {
  const [profile, setProfile] = useState<DatasetProfile | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setProfile(null)
    setError(null)
    getDataset(datasetId)
      .then((p) => {
        if (cancelled) return
        setProfile(p)
        onProfileLoaded?.(p)
      })
      .catch((e) => {
        if (cancelled) return
        setError(e instanceof ApiError ? e.message : 'Could not load dataset profile.')
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId])

  if (error) return <p className="text-destructive text-sm">{error}</p>
  if (!profile) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-1/3" />
        <Skeleton className="h-24 w-full" />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2 text-xs">
        <Badge variant="secondary">{profile.rows.toLocaleString()} rows</Badge>
        <Badge variant="secondary">{profile.columns} columns</Badge>
        <Badge variant="secondary">{profile.duplicate_rows} duplicate rows</Badge>
        <Badge variant="secondary">{(profile.memory_usage_bytes / 1024).toFixed(1)} KB</Badge>
      </div>
      <div className="max-h-64 overflow-y-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Column</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Missing</TableHead>
              <TableHead>Unique</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {profile.column_profiles.map((col) => (
              <TableRow key={col.name}>
                <TableCell className="font-mono text-xs">{col.name}</TableCell>
                <TableCell>
                  <Badge variant="outline">{col.dtype}</Badge>
                </TableCell>
                <TableCell className="text-xs">{col.missing_percentage.toFixed(1)}%</TableCell>
                <TableCell className="text-xs">{col.unique_percentage.toFixed(1)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
