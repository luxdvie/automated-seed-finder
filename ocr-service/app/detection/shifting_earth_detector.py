"""Shifting Earth event detection using template matching."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

from .template_matcher import TemplateMatcher
from ..config import settings

logger = logging.getLogger(__name__)


# Shifting Earth types and their expected regions (in 1000x1000 coordinate system)
SHIFTING_EARTH_REGIONS = {
    "MountainTop": {
        "x_start": 0.0,
        "x_end": 0.4,
        "y_start": 0.0,
        "y_end": 0.4,
        "description": "Large white embedded section in top-left"
    },
    "Crater": {
        "x_start": 0.3,
        "x_end": 0.7,
        "y_start": 0.0,
        "y_end": 0.35,
        "description": "Large red/volcano embedded section in middle-top"
    },
    "Noklateo": {
        "x_start": 0.0,
        "x_end": 0.4,
        "y_start": 0.6,
        "y_end": 1.0,
        "description": "Large blue/silver embedded section in bottom-left"
    },
    "RottedWoods": {
        "x_start": 0.6,
        "x_end": 1.0,
        "y_start": 0.6,
        "y_end": 1.0,
        "description": "Large pink/barren embedded section in bottom-right"
    },
    "GreatHollow": {
        "x_start": 0.0,
        "x_end": 1.0,
        "y_start": 0.0,
        "y_end": 1.0,
        "description": "Entirely different map layout"
    }
}

# App IDs for shifting earth events (matching the frontend)
SHIFTING_EARTH_MAPPING = {
    "MountainTop": "mountaintop",
    "Crater": "crater",
    "Noklateo": "noklateo",
    "RottedWoods": "rotted",
    "GreatHollow": "greatHollow"
}


class ShiftingEarthDetector:
    """Detects Shifting Earth events on the in-game map."""

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize Shifting Earth detector.

        Args:
            templates_dir: Directory containing template images.
        """
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent.parent.parent / "templates"
        self.matcher = TemplateMatcher(str(self.templates_dir))
        self.templates_loaded = False
        self.match_threshold = 0.85  # High threshold to avoid false positives

    def load_templates(self) -> bool:
        """Load Shifting Earth templates.

        Returns:
            True if templates loaded successfully.
        """
        count = self.matcher.load_templates_from_directory("shifting_earth")
        self.templates_loaded = count > 0

        if not self.templates_loaded:
            logger.warning("No Shifting Earth templates found. Detection will be limited.")
        else:
            logger.info(f"Loaded {count} Shifting Earth templates")

        return self.templates_loaded

    def detect(
        self,
        map_region: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[Dict]:
        """Detect Shifting Earth event on the map.

        Args:
            map_region: Extracted map region image.
            threshold: Detection confidence threshold.

        Returns:
            Detection result with event type and confidence, or None.
        """
        if not self.templates_loaded:
            logger.warning("Templates not loaded. Call load_templates() first.")
            return None

        threshold = threshold if threshold is not None else self.match_threshold
        height, width = map_region.shape[:2]

        best_match = None
        best_confidence = 0

        for event_name, region_info in SHIFTING_EARTH_REGIONS.items():
            template_name = f"shifting_earth/{event_name}"

            if template_name not in self.matcher.templates:
                continue

            # Extract the expected region for this event type
            x1 = int(region_info["x_start"] * width)
            x2 = int(region_info["x_end"] * width)
            y1 = int(region_info["y_start"] * height)
            y2 = int(region_info["y_end"] * height)

            search_region = map_region[y1:y2, x1:x2]

            if search_region.size == 0:
                continue

            # Try template matching in this region
            result = self.matcher.match_template_multi_scale(
                search_region,
                template_name,
                scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                threshold=0.0  # Get score even if below threshold
            )

            if result:
                x, y, confidence, scale = result

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        "event": SHIFTING_EARTH_MAPPING.get(event_name, event_name),
                        "event_name": event_name,
                        "confidence": confidence,
                        "scale": scale,
                        "region": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    }

        if best_match and best_match["confidence"] >= threshold:
            return best_match

        return None

    def detect_all(
        self,
        map_region: np.ndarray,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Detect all potential Shifting Earth events (for debugging).

        Args:
            map_region: Extracted map region image.
            threshold: Detection confidence threshold.

        Returns:
            List of all detection results sorted by confidence.
        """
        if not self.templates_loaded:
            return []

        threshold = threshold if threshold is not None else 0.0  # Return all for debugging
        height, width = map_region.shape[:2]

        results = []

        for event_name, region_info in SHIFTING_EARTH_REGIONS.items():
            template_name = f"shifting_earth/{event_name}"

            if template_name not in self.matcher.templates:
                results.append({
                    "event": SHIFTING_EARTH_MAPPING.get(event_name, event_name),
                    "event_name": event_name,
                    "confidence": 0,
                    "error": "Template not loaded"
                })
                continue

            # Extract the expected region
            x1 = int(region_info["x_start"] * width)
            x2 = int(region_info["x_end"] * width)
            y1 = int(region_info["y_start"] * height)
            y2 = int(region_info["y_end"] * height)

            search_region = map_region[y1:y2, x1:x2]

            if search_region.size == 0:
                continue

            result = self.matcher.match_template_multi_scale(
                search_region,
                template_name,
                scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                threshold=0.0
            )

            if result:
                x, y, confidence, scale = result
                results.append({
                    "event": SHIFTING_EARTH_MAPPING.get(event_name, event_name),
                    "event_name": event_name,
                    "confidence": confidence,
                    "scale": scale,
                    "region": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        # Sort by confidence
        results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return results
