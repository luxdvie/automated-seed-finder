"""WebSocket connection manager."""

import asyncio
import json
from typing import Dict, List, Optional, Any
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket to accept.
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: The WebSocket to remove.
        """
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(
        self,
        message: Dict[str, Any],
        websocket: WebSocket
    ) -> bool:
        """Send a message to a specific client.

        Args:
            message: Message dictionary to send.
            websocket: Target WebSocket.

        Returns:
            True if sent successfully, False otherwise.
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message: Message dictionary to broadcast.
        """
        disconnected = []

        async with self._lock:
            connections = self.active_connections.copy()

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)

    async def send_detection_result(
        self,
        websocket: WebSocket,
        nightlord: Optional[str],
        nightlord_confidence: float,
        spawn_slot: Optional[str],
        spawn_confidence: float,
        buildings: List[Dict],
        timestamp: Optional[float] = None
    ) -> bool:
        """Send a detection result to a client.

        Args:
            websocket: Target WebSocket.
            nightlord: Detected nightlord ID or None.
            nightlord_confidence: Confidence score for nightlord.
            spawn_slot: Detected spawn slot ID or None.
            spawn_confidence: Confidence score for spawn.
            buildings: List of detected buildings.
            timestamp: Optional timestamp.

        Returns:
            True if sent successfully.
        """
        import time

        message = {
            "type": "detection_result",
            "data": {
                "timestamp": timestamp or time.time(),
                "nightlord": nightlord,
                "nightlord_confidence": nightlord_confidence,
                "spawn_slot": spawn_slot,
                "spawn_confidence": spawn_confidence,
                "buildings": buildings
            }
        }

        return await self.send_personal_message(message, websocket)

    async def send_status(
        self,
        websocket: WebSocket,
        status: str,
        message: str = "",
        data: Optional[Dict] = None
    ) -> bool:
        """Send a status message to a client.

        Args:
            websocket: Target WebSocket.
            status: Status type (e.g., "connected", "error", "capturing").
            message: Optional status message.
            data: Optional additional data.

        Returns:
            True if sent successfully.
        """
        msg = {
            "type": "status",
            "data": {
                "status": status,
                "message": message,
                **(data or {})
            }
        }

        return await self.send_personal_message(msg, websocket)

    async def send_error(
        self,
        websocket: WebSocket,
        error: str,
        code: Optional[str] = None
    ) -> bool:
        """Send an error message to a client.

        Args:
            websocket: Target WebSocket.
            error: Error description.
            code: Optional error code.

        Returns:
            True if sent successfully.
        """
        message = {
            "type": "error",
            "data": {
                "error": error,
                "code": code
            }
        }

        return await self.send_personal_message(message, websocket)

    @property
    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def is_connected(self, websocket: WebSocket) -> bool:
        """Check if a WebSocket is connected."""
        return websocket in self.active_connections
