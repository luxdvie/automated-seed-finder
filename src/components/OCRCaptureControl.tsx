'use client'

import { useState, useCallback } from 'react'
import { useOCRCapture, DetectionResult } from '@/hooks/useOCRCapture'

interface OCRCaptureControlProps {
  onDetection: (result: DetectionResult) => void
  onNightlordDetected?: (nightlordId: string) => void
  onSpawnDetected?: (slotId: string) => void
  onBuildingDetected?: (slotId: string, buildingType: string) => void
  className?: string
}

export default function OCRCaptureControl({
  onDetection,
  onNightlordDetected,
  onSpawnDetected,
  onBuildingDetected,
  className = '',
}: OCRCaptureControlProps) {
  const [showSettings, setShowSettings] = useState(false)
  const [deviceIndex, setDeviceIndex] = useState(0)
  const [autoApply, setAutoApply] = useState(true)
  const [minConfidence, setMinConfidence] = useState(0.75)

  const handleDetection = useCallback((result: DetectionResult) => {
    onDetection(result)

    if (autoApply) {
      // Auto-apply detected nightlord
      if (result.nightlord && result.nightlord_confidence >= minConfidence) {
        onNightlordDetected?.(result.nightlord)
      }

      // Auto-apply detected spawn
      if (result.spawn_slot && result.spawn_confidence >= minConfidence) {
        onSpawnDetected?.(result.spawn_slot)
      }

      // Auto-apply detected buildings
      result.buildings.forEach((building) => {
        if (building.confidence >= minConfidence) {
          onBuildingDetected?.(building.slot_id, building.building_type)
        }
      })
    }
  }, [autoApply, minConfidence, onDetection, onNightlordDetected, onSpawnDetected, onBuildingDetected])

  const {
    isConnected,
    isCapturing,
    status,
    lastDetection,
    error,
    connect,
    disconnect,
    startCapture,
    stopCapture,
    captureScreenshot,
  } = useOCRCapture({
    minConfidence,
    onDetection: handleDetection,
  })

  const getStatusColor = () => {
    switch (status.status) {
      case 'connected':
        return 'bg-green-500'
      case 'capturing':
        return 'bg-blue-500'
      case 'error':
        return 'bg-red-500'
      default:
        return 'bg-gray-500'
    }
  }

  return (
    <div className={`bg-black bg-opacity-80 rounded-lg p-3 ${className}`}>
      {/* Status indicator */}
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
        <span className="text-xs text-white">
          {status.status === 'disconnected' ? 'OCR Disconnected' :
           status.status === 'connected' ? 'OCR Connected' :
           status.status === 'capturing' ? 'Capturing...' :
           status.status === 'error' ? 'Error' : status.status}
        </span>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="ml-auto text-xs text-gray-400 hover:text-white"
        >
          {showSettings ? '▼' : '▶'}
        </button>
      </div>

      {/* Main controls */}
      <div className="flex gap-2">
        {!isConnected ? (
          <button
            onClick={connect}
            className="flex-1 px-3 py-1 text-xs bg-green-600 hover:bg-green-700 text-white rounded"
          >
            Connect
          </button>
        ) : (
          <>
            {!isCapturing ? (
              <button
                onClick={() => startCapture(deviceIndex)}
                className="flex-1 px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
              >
                Start Capture
              </button>
            ) : (
              <>
                <button
                  onClick={captureScreenshot}
                  className="flex-1 px-3 py-1 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded"
                >
                  Capture
                </button>
                <button
                  onClick={stopCapture}
                  className="px-3 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded"
                >
                  Stop
                </button>
              </>
            )}
            <button
              onClick={disconnect}
              className="px-3 py-1 text-xs bg-gray-600 hover:bg-gray-700 text-white rounded"
            >
              Disconnect
            </button>
          </>
        )}
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs text-gray-400">Device Index</label>
              <input
                type="number"
                min="0"
                max="9"
                value={deviceIndex}
                onChange={(e) => setDeviceIndex(parseInt(e.target.value) || 0)}
                className="w-16 px-2 py-1 text-xs bg-gray-800 text-white rounded"
                disabled={isCapturing}
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-xs text-gray-400">Min Confidence</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={minConfidence}
                onChange={(e) => setMinConfidence(parseFloat(e.target.value) || 0.75)}
                className="w-16 px-2 py-1 text-xs bg-gray-800 text-white rounded"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-xs text-gray-400">Auto-apply detections</label>
              <input
                type="checkbox"
                checked={autoApply}
                onChange={(e) => setAutoApply(e.target.checked)}
                className="w-4 h-4"
              />
            </div>
          </div>
        </div>
      )}

      {/* Last detection info */}
      {lastDetection && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          <div className="text-xs text-gray-400 space-y-1">
            {lastDetection.nightlord && (
              <div className="flex justify-between">
                <span>Nightlord:</span>
                <span className="text-white">
                  {lastDetection.nightlord} ({(lastDetection.nightlord_confidence * 100).toFixed(0)}%)
                </span>
              </div>
            )}
            {lastDetection.spawn_slot && (
              <div className="flex justify-between">
                <span>Spawn:</span>
                <span className="text-white">
                  Slot {lastDetection.spawn_slot} ({(lastDetection.spawn_confidence * 100).toFixed(0)}%)
                </span>
              </div>
            )}
            {lastDetection.buildings.length > 0 && (
              <div>
                <span>Buildings: {lastDetection.buildings.length} detected</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="mt-2 p-2 bg-red-900 bg-opacity-50 rounded text-xs text-red-300">
          {error}
        </div>
      )}
    </div>
  )
}
