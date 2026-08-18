import { Route, Routes } from 'react-router-dom'
import { HomePage } from '@/pages/HomePage'
import { RunPage } from '@/pages/RunPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/runs/:runId" element={<RunPage />} />
    </Routes>
  )
}

export default App
