import { CheckCircle2, XCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import type { RunResultResponse } from '@/lib/types'
import { FailurePanel } from './FailurePanel'
import { ModelComparisonChart } from './ModelComparisonChart'
import { BaselinePanel } from './BaselinePanel'
import { ValidationChecksPanel } from './ValidationChecksPanel'
import { ReproducibilityPanel } from './ReproducibilityPanel'

interface ResultSummaryProps {
  result: RunResultResponse
}

export function ResultSummary({ result }: ResultSummaryProps) {
  const isSuccess = result.status === 'completed'

  return (
    <div className="flex flex-col gap-4">
      <Alert variant={isSuccess ? 'success' : 'destructive'}>
        {isSuccess ? <CheckCircle2 /> : <XCircle />}
        <AlertTitle>{isSuccess ? 'Pipeline completed and passed validation' : 'Pipeline did not complete successfully'}</AlertTitle>
        {result.error && <AlertDescription>{result.error}</AlertDescription>}
      </Alert>

      {result.failure && <FailurePanel failure={result.failure} />}

      <Tabs defaultValue="models">
        <TabsList>
          <TabsTrigger value="models">Model comparison</TabsTrigger>
          <TabsTrigger value="baseline" disabled={!result.baseline}>
            Baseline
          </TabsTrigger>
          <TabsTrigger value="validation" disabled={!result.validation}>
            Guardrails
          </TabsTrigger>
          <TabsTrigger value="reproducibility" disabled={!result.reproducibility}>
            Reproducibility
          </TabsTrigger>
        </TabsList>

        <TabsContent value="models">
          <Card>
            <CardHeader>
              <CardTitle>Trained candidates</CardTitle>
            </CardHeader>
            <CardContent>
              {result.comparison ? (
                <ModelComparisonChart comparison={result.comparison} />
              ) : (
                <p className="text-muted-foreground text-sm">No models were trained on the final attempt.</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="baseline">
          <Card>
            <CardHeader>
              <CardTitle>Trivial-baseline gate</CardTitle>
            </CardHeader>
            <CardContent>{result.baseline && <BaselinePanel baseline={result.baseline} />}</CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="validation">
          <Card>
            <CardHeader>
              <CardTitle>Deterministic guardrails</CardTitle>
            </CardHeader>
            <CardContent>{result.validation && <ValidationChecksPanel validation={result.validation} />}</CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reproducibility">
          <Card>
            <CardHeader>
              <CardTitle>Run reproducibility</CardTitle>
            </CardHeader>
            <CardContent>
              {result.reproducibility && <ReproducibilityPanel reproducibility={result.reproducibility} />}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
