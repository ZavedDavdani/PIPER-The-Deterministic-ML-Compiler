import type { ReproducibilityMetadata } from '@/lib/types'

interface ReproducibilityPanelProps {
  reproducibility: ReproducibilityMetadata
}

export function ReproducibilityPanel({ reproducibility }: ReproducibilityPanelProps) {
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-3">
      <div>
        <div className="text-muted-foreground">Python</div>
        <div className="font-mono">{reproducibility.environment.python_version.split(' ')[0]}</div>
      </div>
      <div>
        <div className="text-muted-foreground">pandas</div>
        <div className="font-mono">{reproducibility.environment.pandas_version}</div>
      </div>
      <div>
        <div className="text-muted-foreground">scikit-learn</div>
        <div className="font-mono">{reproducibility.environment.sklearn_version}</div>
      </div>
      <div>
        <div className="text-muted-foreground">Split random_state</div>
        <div className="font-mono">{reproducibility.split_random_state}</div>
      </div>
      <div>
        <div className="text-muted-foreground">Model random_state</div>
        <div className="font-mono">{reproducibility.model_random_state}</div>
      </div>
      <div className="col-span-2 sm:col-span-3">
        <div className="text-muted-foreground">Dataset fingerprint</div>
        <div className="truncate font-mono" title={reproducibility.dataset_fingerprint}>
          {reproducibility.dataset_fingerprint}
        </div>
      </div>
    </div>
  )
}
