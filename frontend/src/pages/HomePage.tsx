import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { DatasetUpload } from '@/features/datasets/DatasetUpload'
import { DatasetList } from '@/features/datasets/DatasetList'
import { DatasetPreview } from '@/features/datasets/DatasetPreview'
import { RunCreateForm } from '@/features/runs/RunCreateForm'
import { OllamaStatusCard } from '@/features/settings/OllamaStatusCard'
import type { DatasetProfile } from '@/lib/types'

export function HomePage() {
  const navigate = useNavigate()
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)
  const [profile, setProfile] = useState<DatasetProfile | null>(null)
  const [refreshToken, setRefreshToken] = useState(0)

  function handleSelectDataset(id: string) {
    setSelectedDatasetId(id)
    setProfile(null)
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-8">
      <header className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-semibold tracking-tight">PIPER</h1>
          <p className="text-muted-foreground text-sm">
            Autonomous ML Pipeline Intelligence Engine — upload a dataset, choose what to predict, and watch PIPER plan,
            validate, and train a model end to end.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/history" className="text-muted-foreground hover:text-foreground text-sm">
            Run history
          </Link>
          <ThemeToggle />
        </div>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Upload a dataset</CardTitle>
              <CardDescription>CSV, TSV, Excel, JSON, Jupyter notebook, or Parquet.</CardDescription>
            </CardHeader>
            <CardContent>
              <DatasetUpload
                onUploaded={(dataset) => {
                  setRefreshToken((t) => t + 1)
                  handleSelectDataset(dataset.dataset_id)
                }}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Your datasets</CardTitle>
              <CardDescription>Select one to configure a run.</CardDescription>
            </CardHeader>
            <CardContent>
              <DatasetList selectedId={selectedDatasetId} onSelect={handleSelectDataset} refreshToken={refreshToken} />
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-6">
          <OllamaStatusCard />

          {selectedDatasetId ? (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Dataset preview</CardTitle>
                  <CardDescription className="font-mono">{selectedDatasetId}</CardDescription>
                </CardHeader>
                <CardContent>
                  <DatasetPreview datasetId={selectedDatasetId} onProfileLoaded={setProfile} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Configure run</CardTitle>
                  <CardDescription>PIPER proposes the plan; deterministic guardrails validate every step.</CardDescription>
                </CardHeader>
                <CardContent>
                  {profile ? (
                    <RunCreateForm
                      datasetId={selectedDatasetId}
                      columns={profile.column_profiles.map((c) => c.name)}
                      onCreated={(runId) => navigate(`/runs/${runId}`)}
                    />
                  ) : (
                    <p className="text-muted-foreground text-sm">Loading dataset columns…</p>
                  )}
                </CardContent>
              </Card>
            </>
          ) : (
            <Card className="border-dashed">
              <CardContent className="text-muted-foreground py-10 text-center text-sm">
                Select or upload a dataset to configure a run.
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
