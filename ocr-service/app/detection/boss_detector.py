"""Boss (Nightlord) detection using template matching."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

from .template_matcher import TemplateMatcher
from ..config import settings, NIGHTLORD_MAPPING, NIGHTLORD_COORDINATE

logger = logging.getLogger(__name__)


class BossDetector:
    """Detects nightlord icons on the in-game map."""

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize boss detector.

        Args:
            templates_dir: Directory containing template images.
        """
        self.matcher = TemplateMatcher(templates_dir)
        self.templates_loaded = False

        # Expected nightlord position in 1000x1000 coordinate system
        self.expected_x = NIGHTLORD_COORDINATE["x"]
        self.expected_y = NIGHTLORD_COORDINATE["y"]

        # Search region around expected position (in pixels after scaling)
        self.search_margin = 100

    def load_templates(self) -> bool:
        """Load nightlord templates.

        Returns:
            True if templates loaded successfully.
        """
        count = self.matcher.load_templates_from_directory("nightlords")
        self.templates_loaded = count > 0

        if not self.templates_loaded:
            logger.warning("No nightlord templates found. Detection will be limited.")

        return self.templates_loaded

    def get_search_region(
        self,
        map_region: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """Calculate the search region for nightlord icon.

        Args:
            map_region: The extracted map region.

        Returns:
            Tuple of (x1, y1, x2, y2) defining the search region.
        """
        height, width = map_region.shape[:2]

        # Scale expected position to pixel coordinates
        scale_x = width / settings.coordinate_system_size
        scale_y = height / settings.coordinate_system_size

        center_x = int(self.expected_x * scale_x)
        center_y = int(self.expected_y * scale_y)

        # Define search region with margin
        x1 = max(0, center_x - self.search_margin)
        y1 = max(0, center_y - self.search_margin)
        x2 = min(width, center_x + self.search_margin)
        y2 = min(height, center_y + self.search_margin)

        return x1, y1, x2, y2

    def detect(
        self,
        map_region: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[Dict]:
        """Detect nightlord on the map.

        Args:
            map_region: Extracted map region image.
            threshold: Detection confidence threshold.

        Returns:
            Detection result with nightlord ID and confidence, or None.
        """
        if not self.templates_loaded:
            logger.warning("Templates not loaded. Call load_templates() first.")
            return None

        threshold = threshold or settings.template_match_threshold

        # Extract search region around expected nightlord position
        x1, y1, x2, y2 = self.get_search_region(map_region)
        search_region = map_region[y1:y2, x1:x2]

        if search_region.size == 0:
            logger.warning("Empty search region")
            return None

        # Find best matching nightlord template
        match = self.matcher.find_best_match(search_region, "nightlords", threshold)

        if match:
            # Map template name to app nightlord ID
            template_name = match["template"]
            nightlord_id = self._map_template_to_id(template_name)

            if nightlord_id:
                return {
                    "nightlord": nightlord_id,
                    "confidence": match["confidence"],
                    "template_name": template_name,
                    "position": {
                        "x": x1 + match["x"],
                        "y": y1 + match["y"]
                    }
                }

        return None

    def _map_template_to_id(self, template_name: str) -> Optional[str]:
        """Map a template filename to the app's nightlord ID.

        Args:
            template_name: Template filename (without extension).

        Returns:
            Nightlord ID (e.g., "1_Gladius") or None.
        """
        # Try direct mapping from NIGHTLORD_MAPPING
        if template_name in NIGHTLORD_MAPPING:
            return NIGHTLORD_MAPPING[template_name]

        # Try if template is already in ID format
        if template_name.startswith(tuple("123456789")):
            return template_name

        # Try case-insensitive match
        template_lower = template_name.lower()
        for game_name, app_id in NIGHTLORD_MAPPING.items():
            if game_name.lower() == template_lower:
                return app_id

        logger.warning(f"Unknown nightlord template: {template_name}")
        return None

    def detect_from_screenshot(
        self,
        screenshot_path: str,
        threshold: Optional[float] = None
    ) -> Optional[Dict]:
        """Detect nightlord from a screenshot file.

        Args:
            screenshot_path: Path to screenshot image.
            threshold: Detection confidence threshold.

        Returns:
            Detection result or None.
        """
        image = cv2.imread(screenshot_path)
        if image is None:
            logger.error(f"Failed to load screenshot: {screenshot_path}")
            return None

        return self.detect(image, threshold)
