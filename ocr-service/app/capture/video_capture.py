"""Video capture module for HDMI capture via OpenCV."""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class VideoCapture:
    """Handles video capture from HDMI capture card."""

    def __init__(self, device_index: Optional[int] = None):
        """Initialize video capture.

        Args:
            device_index: Camera/capture device index. Defaults to config value.
        """
        self.device_index = device_index if device_index is not None else settings.capture_device_index
        self.capture: Optional[cv2.VideoCapture] = None
        self._is_capturing = False

    def start(self) -> bool:
        """Start video capture.

        Returns:
            True if capture started successfully, False otherwise.
        """
        if self._is_capturing:
            logger.warning("Capture already running")
            return True

        try:
            # On macOS, use AVFoundation backend
            self.capture = cv2.VideoCapture(self.device_index, cv2.CAP_AVFOUNDATION)

            if not self.capture.isOpened():
                # Fallback to default backend
                self.capture = cv2.VideoCapture(self.device_index)

            if not self.capture.isOpened():
                logger.error(f"Failed to open capture device {self.device_index}")
                return False

            # Set capture properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.capture_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.capture_height)
            self.capture.set(cv2.CAP_PROP_FPS, settings.capture_fps)

            self._is_capturing = True
            logger.info(f"Started capture on device {self.device_index}")
            return True

        except Exception as e:
            logger.error(f"Error starting capture: {e}")
            return False

    def stop(self) -> None:
        """Stop video capture and release resources."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self._is_capturing = False
        logger.info("Capture stopped")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the capture device.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if not self._is_capturing or self.capture is None:
            return False, None

        ret, frame = self.capture.read()
        if not ret:
            logger.warning("Failed to read frame")
            return False, None

        return True, frame

    def capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture a single screenshot.

        Returns:
            Frame as numpy array, or None if capture failed.
        """
        success, frame = self.read_frame()
        return frame if success else None

    def get_frame_size(self) -> Tuple[int, int]:
        """Get the current frame size.

        Returns:
            Tuple of (width, height).
        """
        if self.capture is not None:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return settings.capture_width, settings.capture_height

    @property
    def is_capturing(self) -> bool:
        """Check if capture is currently running."""
        return self._is_capturing

    @staticmethod
    def list_devices() -> list[dict]:
        """List available capture devices.

        Returns:
            List of device info dictionaries.
        """
        devices = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append({
                    "index": i,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                })
                cap.release()
        return devices

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
