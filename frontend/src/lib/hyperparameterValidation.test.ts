import { describe, expect, it } from 'vitest'
import {
  defaultValueForHyperparameter,
  hyperparametersForAlgorithm,
  validateHyperparameterValue,
} from '@/lib/hyperparameterValidation'

describe('hyperparameterValidation', () => {
  const nEstimators = hyperparametersForAlgorithm('random_forest').find(
    (item) => item.name === 'n_estimators',
  )!

  it('accepts valid integer hyperparameter values', () => {
    expect(validateHyperparameterValue(nEstimators, '100')).toEqual({ valid: true, value: 100 })
    expect(validateHyperparameterValue(nEstimators, '50')).toEqual({ valid: true, value: 50 })
    expect(validateHyperparameterValue(nEstimators, '200')).toEqual({ valid: true, value: 200 })
  })

  it('rejects decimal values for integer hyperparameters', () => {
    const result = validateHyperparameterValue(nEstimators, '0.1')
    expect(result.valid).toBe(false)
    if (!result.valid) {
      expect(result.message).toBe('n_estimators must be a whole number.')
    }
  })

  it('rejects non-numeric values', () => {
    const result = validateHyperparameterValue(nEstimators, 'abc')
    expect(result.valid).toBe(false)
    if (!result.valid) {
      expect(result.message).toContain('valid value')
    }
  })

  it('allows float values for regularization parameters', () => {
    const cParam = hyperparametersForAlgorithm('logistic_regression').find((item) => item.name === 'C')!
    expect(validateHyperparameterValue(cParam, '0.1')).toEqual({ valid: true, value: 0.1 })
  })

  it('provides algorithm-specific defaults', () => {
    expect(defaultValueForHyperparameter(nEstimators)).toBe('100')
  })
})
