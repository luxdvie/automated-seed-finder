'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

export interface DetectionResult {
  timestamp: number
  nightlord: string | null
  nightlord_confidence: number
  spawn_slot: string | null
  spawn_confidence: number
  buildings: Array<{
    slot_id: string
    building_type: string
    confidence: number
  }>
}

export interface OCRStatus {
  status: 'disconnected' | 'connected' | 'capturing' | 'stopped' | 'error'
  message: string
}

export interface UseOCRCaptureOptions {
  wsUrl?: string
  autoConnect?: boolean
  minConfidence?: number
  onDetection?: (result: DetectionResult) => void
  onStatusChange?: (status: OCRStatus) => void
  onError?: (error: string) => void
}

export interface UseOCRCaptureReturn {
  isConnected: boolean
  isCapturing: boolean
  status: OCRStatus
  lastDetection: DetectionResult | null
  error: string | null
  connect: () => void
  disconnect: () => void
  startCapture: (deviceIndex?: number) => void
  stopCapture: () => void
  captureScreenshot: () => void
}

const DEFAULT_WS_URL = 'ws://localhost:8000/ws'
const RECONNECT_DELAY = 3000
const PING_INTERVAL = 30000

export function useOCRCapture(options: UseOCRCaptureOptions = {}): UseOCRCaptureReturn {
  const {
    wsUrl = DEFAULT_WS_URL,
    autoConnect = false,
    minConfidence = 0.75,
    onDetection,
    onStatusChange,
    onError,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [isCapturing, setIsCapturing] = useState(false)
  const [status, setStatus] = useState<OCRStatus>({ status: 'disconnected', message: '' })
  const [lastDetection, setLastDetection] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const updateStatus = useCallback((newStatus: OCRStatus) => {
    setStatus(newStatus)
    onStatusChange?.(newStatus)
  }, [onStatusChange])

  const handleError = useCallback((errorMsg: string) => {
    setError(errorMsg)
    onError?.(errorMsg)
  }, [onError])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close()
    }

    try {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        setIsConnected(true)
        setError(null)
        updateStatus({ status: 'connected', message: 'Connected to OCR service' })

        // Start ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current)
        }
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }))
          }
        }, PING_INTERVAL)
      }

      ws.onclose = () => {
        setIsConnected(false)
        setIsCapturing(false)
        updateStatus({ status: 'disconnected', message: 'Disconnected from OCR service' })

        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current)
          pingIntervalRef.current = null
        }
      }

      ws.onerror = () => {
        handleError('WebSocket connection error')
        updateStatus({ status: 'error', message: 'Connection error' })
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          handleMessage(message)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      wsRef.current = ws
    } catch (e) {
      handleError(`Failed to connect: ${e}`)
    }
  }, [wsUrl, updateStatus, handleError])

  const handleMessage = useCallback((message: { type: string; data: Record<string, unknown> }) => {
    switch (message.type) {
      case 'status':
        {
          const statusData = message.data as { status: string; message: string }
          const newStatus: OCRStatus = {
            status: statusData.status as OCRStatus['status'],
            message: statusData.message,
          }
          updateStatus(newStatus)

          if (statusData.status === 'capturing') {
            setIsCapturing(true)
          } else if (statusData.status === 'stopped') {
            setIsCapturing(false)
          }
        }
        break

      case 'detection_result':
        {
          const detection = message.data as unknown as DetectionResult

          // Filter by confidence threshold
          const filteredDetection: DetectionResult = {
            ...detection,
            nightlord: detection.nightlord_confidence >= minConfidence ? detection.nightlord : null,
            spawn_slot: detection.spawn_confidence >= minConfidence ? detection.spawn_slot : null,
            buildings: detection.buildings.filter((b) => b.confidence >= minConfidence),
          }

          setLastDetection(filteredDetection)
          onDetection?.(filteredDetection)
        }
        break

      case 'error':
        {
          const errorData = message.data as { error: string }
          handleError(errorData.error)
        }
        break

      case 'pong':
        // Keep-alive response, no action needed
        break

      default:
        console.warn('Unknown message type:', message.type)
    }
  }, [minConfidence, updateStatus, handleError, onDetection])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setIsConnected(false)
    setIsCapturing(false)
    updateStatus({ status: 'disconnected', message: 'Disconnected' })
  }, [updateStatus])

  const sendMessage = useCallback((type: string, data?: Record<string, unknown>) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      handleError('Not connected to OCR service')
      return false
    }

    try {
      wsRef.current.send(JSON.stringify({ type, data }))
      return true
    } catch (e) {
      handleError(`Failed to send message: ${e}`)
      return false
    }
  }, [handleError])

  const startCapture = useCallback((deviceIndex = 0) => {
    sendMessage('start_capture', { device_index: deviceIndex })
  }, [sendMessage])

  const stopCapture = useCallback(() => {
    sendMessage('stop_capture')
  }, [sendMessage])

  const captureScreenshot = useCallback(() => {
    sendMessage('capture_screenshot')
  }, [sendMessage])

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [autoConnect, connect, disconnect])

  return {
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
  }
}
