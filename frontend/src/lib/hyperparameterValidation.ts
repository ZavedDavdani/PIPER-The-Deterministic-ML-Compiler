/**
 * Client-side hyperparameter validation aligned with backend training.py bounds.
 */

export type AlgorithmName = 'logistic_regression' | 'random_forest'

export type HyperparamValueType = 'integer' | 'float'

export interface HyperparameterMeta {
  name: string
  label: string
  valueType: HyperparamValueType
  min: number
  max: number
  algorithms: AlgorithmName[]
  placeholder: string
}

export const HYPERPARAMETER_CATALOG: HyperparameterMeta[] = [
  {
    name: 'C',
    label: 'Regularization Strength (C)',
    valueType: 'float',
    min: 0.001,
    max: 100,
    algorithms: ['logistic_regression'],
    placeholder: 'e.g. 0.1, 1.0',
  },
  {
    name: 'max_iter',
    label: 'Max Iterations (max_iter)',
    valueType: 'integer',
    min: 100,
    max: 5000,
    algorithms: ['logistic_regression'],
    placeholder: 'e.g. 1000',
  },
  {
    name: 'n_estimators',
    label: 'Tree Count (n_estimators)',
    valueType: 'integer',
    min: 50,
    max: 500,
    algorithms: ['random_forest'],
    placeholder: 'e.g. 50, 100, 200',
  },
  {
    name: 'max_depth',
    label: 'Max Depth (max_depth)',
    valueType: 'integer',
    min: 1,
    max: 50,
    algorithms: ['random_forest'],
    placeholder: 'e.g. 5, 10',
  },
  {
    name: 'min_samples_split',
    label: 'Min Samples Split (min_samples_split)',
    valueType: 'integer',
    min: 2,
    max: 20,
    algorithms: ['random_forest'],
    placeholder: 'e.g. 2, 5',
  },
  {
    name: 'min_samples_leaf',
    label: 'Min Samples Leaf (min_samples_leaf)',
    valueType: 'integer',
    min: 1,
    max: 20,
    algorithms: ['random_forest'],
    placeholder: 'e.g. 1, 2',
  },
]

export function hyperparametersForAlgorithm(algorithm: AlgorithmName): HyperparameterMeta[] {
  return HYPERPARAMETER_CATALOG.filter((item) => item.algorithms.includes(algorithm))
}

export function defaultHyperparameterForAlgorithm(algorithm: AlgorithmName): HyperparameterMeta {
  const options = hyperparametersForAlgorithm(algorithm)
  return options[0]
}

export function defaultValueForHyperparameter(meta: HyperparameterMeta): string {
  if (meta.name === 'C') return '1.0'
  if (meta.name === 'max_iter') return '1000'
  if (meta.name === 'n_estimators') return '100'
  if (meta.name === 'max_depth') return '10'
  if (meta.name === 'min_samples_split') return '2'
  if (meta.name === 'min_samples_leaf') return '1'
  return String(meta.min)
}

export type HyperparameterValidationResult =
  | { valid: true; value: number }
  | { valid: false; message: string }

export function validateHyperparameterValue(
  meta: HyperparameterMeta,
  rawValue: string,
): HyperparameterValidationResult {
  const trimmed = rawValue.trim()
  if (!trimmed) {
    return { valid: false, message: `Please enter a valid value for ${meta.label}.` }
  }

  const parsed = Number(trimmed)
  if (!Number.isFinite(parsed)) {
    return { valid: false, message: `Please enter a valid value for ${meta.label}.` }
  }

  if (meta.valueType === 'integer') {
    if (!Number.isInteger(parsed)) {
      return { valid: false, message: `${meta.name} must be a whole number.` }
    }
  }

  if (parsed < meta.min || parsed > meta.max) {
    return {
      valid: false,
      message: `${meta.name} must be between ${meta.min} and ${meta.max}.`,
    }
  }

  return { valid: true, value: parsed }
}
