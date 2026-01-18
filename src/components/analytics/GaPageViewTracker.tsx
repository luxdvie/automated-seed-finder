'use client'

import { useEffect, useRef } from 'react'
import { usePathname } from 'next/navigation'

type GtagFunction = (command: 'event' | 'config' | 'js', target: string | Date, params?: Record<string, unknown>) => void

type WindowWithGtag = Window & {
  gtag?: GtagFunction
}

function getMeasurementId(): string | null {
  const value = process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID
  if (!value) return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function buildPageLocation(): string | null {
  if (typeof window === 'undefined') return null
  const href = window.location.href
  return href && href.length > 0 ? href : null
}

function buildPageTitle(): string | null {
  if (typeof document === 'undefined') return null
  const title = document.title
  return title && title.length > 0 ? title : null
}

function buildPagePathFromWindow(pathnameFallback: string): string {
  if (typeof window === 'undefined') return pathnameFallback
  const path = window.location.pathname
  const search = window.location.search
  if (path && search && search.length > 0) return `${path}${search}`
  return path && path.length > 0 ? path : pathnameFallback
}

export function GaPageViewTracker() {
  const pathname = usePathname()
  const lastTrackedPathnameRef = useRef<string | null>(null)
  const timeoutIdRef = useRef<number | null>(null)

  useEffect(() => {
    if (process.env.NODE_ENV !== 'production') return
    if (!getMeasurementId()) return

    if (lastTrackedPathnameRef.current === pathname) return
    lastTrackedPathnameRef.current = pathname

    const windowWithGtag = window as WindowWithGtag

    const pagePath = buildPagePathFromWindow(pathname)
    const pageLocation = buildPageLocation()
    const pageTitle = buildPageTitle()

    const params: Record<string, unknown> = {
      page_path: pagePath,
    }

    if (pageLocation) params.page_location = pageLocation
    if (pageTitle) params.page_title = pageTitle

    let isCanceled = false

    const clearExistingTimeout = () => {
      if (timeoutIdRef.current === null) return
      window.clearTimeout(timeoutIdRef.current)
      timeoutIdRef.current = null
    }

    const trySendPageView = (attempt: number) => {
      if (isCanceled) return

      const gtag = windowWithGtag.gtag
      if (typeof gtag === 'function') {
        clearExistingTimeout()
        gtag('event', 'page_view', params)
        return
      }

      if (attempt >= 20) {
        clearExistingTimeout()
        return
      }

      clearExistingTimeout()
      timeoutIdRef.current = window.setTimeout(() => {
        trySendPageView(attempt + 1)
      }, 100)
    }

    trySendPageView(0)

    return () => {
      isCanceled = true
      clearExistingTimeout()
    }
  }, [pathname])

  return null
}
