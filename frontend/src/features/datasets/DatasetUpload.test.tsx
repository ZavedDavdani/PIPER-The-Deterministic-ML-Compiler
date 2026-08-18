import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { DatasetUpload } from './DatasetUpload'
import { server } from '@/test/mswServer'
import { API_BASE_URL } from '@/lib/api'
import { fixtureDatasetProfile } from '@/test/fixtures'

describe('DatasetUpload', () => {
  it('uploads a selected CSV and calls onUploaded with the real response', async () => {
    const onUploaded = vi.fn()
    render(<DatasetUpload onUploaded={onUploaded} />)

    const file = new File(['a,b\n1,2\n'], 'telco.csv', { type: 'text/csv' })
    const input = screen.getByLabelText(/choose dataset file/i)
    await userEvent.upload(input, file)

    await waitFor(() => expect(onUploaded).toHaveBeenCalledTimes(1))
    expect(onUploaded).toHaveBeenCalledWith(
      expect.objectContaining({ dataset_id: fixtureDatasetProfile.dataset_id }),
    )
  })

  it.each(['data.tsv', 'book.xlsx', 'data.json', 'analysis.ipynb', 'table.parquet'])(
    'accepts %s (multi-format ingestion) and calls the API',
    async (filename) => {
      const onUploaded = vi.fn()
      render(<DatasetUpload onUploaded={onUploaded} />)

      const file = new File(['payload'], filename, { type: 'application/octet-stream' })
      await userEvent.upload(screen.getByLabelText(/choose dataset file/i), file)

      await waitFor(() => expect(onUploaded).toHaveBeenCalledTimes(1))
    },
  )

  it('rejects a genuinely unsupported file dropped onto the dropzone without calling the API', async () => {
    // Uses drag-and-drop (not the file-picker input) — the picker's
    // `accept` list already keeps the OS file dialog from offering
    // unsupported files at all, but drag-and-drop bypasses that, which
    // is exactly why DatasetUpload has its own explicit extension check.
    const onUploaded = vi.fn()
    render(<DatasetUpload onUploaded={onUploaded} />)

    const file = new File(['not tabular'], 'notes.txt', { type: 'text/plain' })
    const dropzone = screen.getByLabelText(/upload a dataset/i)
    fireEvent.drop(dropzone, { dataTransfer: { files: [file] } })

    expect(await screen.findByText(/unsupported file type/i)).toBeInTheDocument()
    expect(onUploaded).not.toHaveBeenCalled()
  })

  it('shows the detected format and dimensions after a successful upload', async () => {
    server.use(
      http.post(`${API_BASE_URL}/datasets`, () =>
        HttpResponse.json(
          {
            dataset_id: 'dataset_xyz',
            filename: 'book.xlsx',
            rows: 1234,
            columns: ['a', 'b', 'c'],
            detected_format: 'excel',
            column_count: 3,
            sheet_name: 'Primary',
            available_sheets: [
              { name: 'Primary', rows: 1234, columns: 3 },
              { name: 'Notes', rows: 2, columns: 1 },
            ],
            notes: ['Workbook contains 2 worksheets; ingested the first non-empty one.'],
          },
          { status: 201 },
        ),
      ),
    )
    render(<DatasetUpload onUploaded={vi.fn()} />)

    const file = new File(['x'], 'book.xlsx', { type: 'application/octet-stream' })
    await userEvent.upload(screen.getByLabelText(/choose dataset file/i), file)

    expect(await screen.findByText(/detected excel/i)).toBeInTheDocument()
    expect(screen.getByText(/1,234 rows × 3 columns/i)).toBeInTheDocument()
    expect(screen.getByText('Primary')).toBeInTheDocument()
    expect(screen.getByText(/2 worksheets/i)).toBeInTheDocument()
  })

  it('shows a real backend error message on upload failure', async () => {
    server.use(
      http.post(`${API_BASE_URL}/datasets`, () =>
        HttpResponse.json({ detail: 'Uploaded CSV has zero columns.' }, { status: 422 }),
      ),
    )
    const onUploaded = vi.fn()
    render(<DatasetUpload onUploaded={onUploaded} />)

    const file = new File([''], 'empty.csv', { type: 'text/csv' })
    const input = screen.getByLabelText(/choose dataset file/i)
    await userEvent.upload(input, file)

    expect(await screen.findByText(/zero columns/i)).toBeInTheDocument()
    expect(onUploaded).not.toHaveBeenCalled()
  })
})
