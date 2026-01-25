"""Frame processor for extracting map region from captured frames."""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Processes captured frames to extract map region."""

    def __init__(self):
        """Initialize frame processor."""
        self.last_frame: Optional[np.ndarray] = None
        self.last_map_region: Optional[np.ndarray] = None

    def extract_map_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract the map overlay region from a frame.

        The map typically appears on the right side of the screen when opened.
        The in-game map is square, so we extract a square region to match
        the 1000x1000 coordinate system.

        Args:
            frame: Full captured frame.

        Returns:
            Cropped map region (square), or None if extraction failed.
        """
        if frame is None:
            return None

        try:
            height, width = frame.shape[:2]

            # Calculate initial crop coordinates based on settings
            y_start = int(height * settings.map_region_y_start)
            y_end = int(height * settings.map_region_y_end)
            region_height = y_end - y_start

            # Make the region square using height as the reference
            # The map is on the right side of the screen
            x_end = int(width * settings.map_region_x_end)
            x_start = x_end - region_height  # Square: width = height

            # Ensure x_start doesn't go negative
            if x_start < 0:
                x_start = 0
                # Adjust y to maintain square aspect if needed
                region_height = x_end - x_start
                y_center = (y_start + y_end) // 2
                y_start = y_center - region_height // 2
                y_end = y_start + region_height

            # Crop the map region (now square)
            map_region = frame[y_start:y_end, x_start:x_end]

            # Trim left padding - the in-game map is right-aligned within the region
            # Only trim left side; coordinate conversion handles non-square regions
            left_trim = settings.map_region_left_trim
            if left_trim > 0 and left_trim < map_region.shape[1]:
                map_region = map_region[:, left_trim:]

            self.last_frame = frame
            self.last_map_region = map_region

            logger.debug(f"Extracted square map region: {map_region.shape[1]}x{map_region.shape[0]}")

            return map_region

        except Exception as e:
            logger.error(f"Error extracting map region: {e}")
            return None

    def is_map_visible(self, frame: np.ndarray) -> bool:
        """Detect if the in-game map overlay is currently visible.

        Uses edge detection and contour analysis to detect map presence.

        Args:
            frame: Full captured frame.

        Returns:
            True if map appears to be visible, False otherwise.
        """
        if frame is None:
            return False

        try:
            # Extract potential map region
            map_region = self.extract_map_region(frame)
            if map_region is None:
                return False

            # Convert to grayscale
            gray = cv2.cvtColor(map_region, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Count edge pixels - map typically has many edges from icons/terrain
            edge_ratio = np.count_nonzero(edges) / edges.size

            # Map is likely visible if there's a reasonable amount of edges
            # This threshold may need tuning based on actual gameplay
            return edge_ratio > 0.02 and edge_ratio < 0.3

        except Exception as e:
            logger.error(f"Error detecting map visibility: {e}")
            return False

    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better template matching.

        Args:
            image: Input image.

        Returns:
            Preprocessed image.
        """
        if len(image.shape) == 3:
            # Convert to grayscale for template matching
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        return blurred

    def scale_to_coordinate_system(
        self,
        x: int,
        y: int,
        region_width: int,
        region_height: int
    ) -> Tuple[int, int]:
        """Scale pixel coordinates to the 1000x1000 coordinate system.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
            region_width: Width of the map region.
            region_height: Height of the map region.

        Returns:
            Tuple of (scaled_x, scaled_y) in 1000x1000 system.
        """
        scale_x = settings.coordinate_system_size / region_width
        scale_y = settings.coordinate_system_size / region_height

        return int(x * scale_x), int(y * scale_y)

    def get_region_dimensions(self, frame: np.ndarray) -> Tuple[int, int]:
        """Get the dimensions of the map region for a given frame.

        Args:
            frame: Full captured frame.

        Returns:
            Tuple of (width, height) of the map region.
        """
        height, width = frame.shape[:2]

        region_width = int(width * (settings.map_region_x_end - settings.map_region_x_start))
        region_height = int(height * (settings.map_region_y_end - settings.map_region_y_start))

        return region_width, region_height
