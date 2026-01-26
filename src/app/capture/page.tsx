'use client'

import { Suspense, useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

// Map shifting earth detection results to URL paths
const SHIFTING_EARTH_TO_PATH: Record<string, string> = {
  'mountaintop': '/map/mountaintop',
  'crater': '/map/crater',
  'noklateo': '/map/noklateo',
  'rotted': '/map/rotted',
  'greatHollow': '/map/greatHollow',
}

interface SpawnDebug {
  pixel: { x: number; y: number }
  system: { x: number; y: number }
}

interface DetectionResult {
  nightlord: string | null
  nightlord_confidence: number
  spawn_slot: string | null
  spawn_confidence: number
  spawn_debug: SpawnDebug | null
  shifting_earth: string | null
  shifting_earth_confidence: number
}

function CapturePageContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [status, setStatus] = useState<'capturing' | 'processing' | 'redirecting' | 'error'>('capturing')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)

  // Check if user is at a church spawn (from Stream Deck button)
  const isChurchSpawn = searchParams.get('church') === 'true'

  useEffect(() => {
    async function captureAndRedirect() {
      try {
        setStatus('capturing')

        // Call the OCR capture endpoint (uses server-configured default monitor)
        // Use the same hostname as the current page, but port 8000
        const ocrHost = typeof window !== 'undefined'
          ? `http://${window.location.hostname}:8000`
          : 'http://localhost:8000'
        const response = await fetch(`${ocrHost}/capture-monitor`)

        if (!response.ok) {
          throw new Error(`Capture failed: ${response.statusText}`)
        }

        const data: DetectionResult = await response.json()
        console.log('OCR response:', data)
        console.log('spawn_slot value:', data.spawn_slot, 'type:', typeof data.spawn_slot)
        setResult(data)
        setStatus('processing')

        // Determine the map path based on shifting earth
        let mapPath = '/map/normal'
        if (data.shifting_earth && SHIFTING_EARTH_TO_PATH[data.shifting_earth]) {
          mapPath = SHIFTING_EARTH_TO_PATH[data.shifting_earth]
        }

        // Build query params for auto-selection
        const params = new URLSearchParams()

        if (data.nightlord) {
          params.set('nightlord', data.nightlord)
        }

        // Pass spawn coordinates so MapBuilder can find nearest VALID spawn
        if (data.spawn_debug?.system) {
          params.set('spawn_x', String(data.spawn_debug.system.x))
          params.set('spawn_y', String(data.spawn_debug.system.y))
        }

        // Pass OCR-detected spawn slot for debugging
        console.log('Setting ocr_slot:', data.spawn_slot)
        if (data.spawn_slot) {
          params.set('ocr_slot', data.spawn_slot)
        }
        console.log('Final params:', params.toString())

        // Pass spawn type (from Stream Deck button: church or empty)
        params.set('spawn_type', isChurchSpawn ? 'church_spawn' : 'empty_spawn')

        // Construct the full URL
        const redirectUrl = `${mapPath}${params.toString() ? '?' + params.toString() : ''}`

        setStatus('redirecting')

        // Small delay to show the results before redirecting
        await new Promise(resolve => setTimeout(resolve, 500))

        router.push(redirectUrl)

      } catch (err) {
        console.error('Capture error:', err)
        setError(err instanceof Error ? err.message : 'Unknown error')
        setStatus('error')
      }
    }

    captureAndRedirect()
  }, [router, isChurchSpawn])

  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
      <div className="text-center p-8 max-w-md">
        <h1 className="text-2xl font-bold mb-6">OCR Capture</h1>

        {status === 'capturing' && (
          <div className="space-y-4">
            <div className="animate-pulse text-xl">Capturing screen...</div>
            <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          </div>
        )}

        {status === 'processing' && result && (
          <div className="space-y-4">
            <div className="text-xl">Processing...</div>
            <div className="text-left bg-gray-800 p-4 rounded text-sm">
              <p><strong>Shifting Earth:</strong> {result.shifting_earth || 'None'}</p>
              <p><strong>Nightlord:</strong> {result.nightlord || 'Not detected'}</p>
              <p><strong>Spawn Slot:</strong> {result.spawn_slot || 'Not detected'}</p>
            </div>
          </div>
        )}

        {status === 'redirecting' && (
          <div className="space-y-4">
            <div className="text-xl text-green-400">Redirecting to map...</div>
          </div>
        )}

        {status === 'error' && (
          <div className="space-y-4">
            <div className="text-xl text-red-400">Error</div>
            <p className="text-gray-400">{error}</p>
            <p className="text-sm text-gray-500">
              Make sure the OCR service is running at localhost:8000
            </p>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default function CapturePage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center p-8">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    }>
      <CapturePageContent />
    </Suspense>
  )
}
