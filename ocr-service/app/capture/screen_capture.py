"""Screen capture module for capturing monitors."""

import mss
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Captures screenshots from monitors."""

    def __init__(self):
        """Initialize screen capture."""
        self.sct = mss.mss()

    def list_monitors(self) -> List[Dict]:
        """List all available monitors.

        Returns:
            List of monitor info dictionaries.
            Index 0 is "all monitors combined", 1+ are individual monitors.
        """
        monitors = []
        for i, monitor in enumerate(self.sct.monitors):
            monitors.append({
                "index": i,
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"],
                "is_combined": i == 0  # Index 0 is all monitors combined
            })
        return monitors

    def capture_monitor(self, monitor_index: int = 1) -> Optional[np.ndarray]:
        """Capture a specific monitor.

        Args:
            monitor_index: Monitor index (1 = first monitor, 2 = second, etc.)
                          Index 0 captures all monitors combined.

        Returns:
            Screenshot as numpy array (BGR format for OpenCV), or None if failed.
        """
        try:
            monitors = self.sct.monitors

            if monitor_index < 0 or monitor_index >= len(monitors):
                logger.error(f"Invalid monitor index: {monitor_index}. Available: 0-{len(monitors)-1}")
                return None

            monitor = monitors[monitor_index]
            screenshot = self.sct.grab(monitor)

            # Convert to numpy array (BGRA format)
            img = np.array(screenshot)

            # Convert BGRA to BGR for OpenCV
            img_bgr = img[:, :, :3]

            return img_bgr

        except Exception as e:
            logger.error(f"Error capturing monitor {monitor_index}: {e}")
            return None

    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int
    ) -> Optional[np.ndarray]:
        """Capture a specific region of the screen.

        Args:
            left: X coordinate of top-left corner.
            top: Y coordinate of top-left corner.
            width: Width of region.
            height: Height of region.

        Returns:
            Screenshot as numpy array (BGR format), or None if failed.
        """
        try:
            region = {"left": left, "top": top, "width": width, "height": height}
            screenshot = self.sct.grab(region)

            img = np.array(screenshot)
            img_bgr = img[:, :, :3]

            return img_bgr

        except Exception as e:
            logger.error(f"Error capturing region: {e}")
            return None


# Global instance for reuse
screen_capture = ScreenCapture()
