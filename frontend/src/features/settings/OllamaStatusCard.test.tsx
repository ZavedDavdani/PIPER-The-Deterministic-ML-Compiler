import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { OllamaStatusCard } from '@/features/settings/OllamaStatusCard'

describe('OllamaStatusCard', () => {
  it('loads provider settings and allows saving an arbitrary model', async () => {
    const user = userEvent.setup()
    render(<OllamaStatusCard />)

    expect(await screen.findByText('LLM Provider')).toBeInTheDocument()
    expect(await screen.findByDisplayValue('gpt-5.6-luna')).toBeInTheDocument()

    const modelInput = screen.getByDisplayValue('gpt-5.6-luna')
    await user.clear(modelInput)
    await user.type(modelInput, 'custom-model:latest')
    await user.click(screen.getByRole('button', { name: /save/i }))

    await waitFor(() => {
      expect(screen.getByDisplayValue('custom-model:latest')).toBeInTheDocument()
    })
  })
})
