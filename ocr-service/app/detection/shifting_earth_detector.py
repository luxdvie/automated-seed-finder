"""Shifting Earth event detection using template matching."""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

from .template_matcher import TemplateMatcher

logger = logging.getLogger(__name__)


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
        self.match_threshold = 0.65  # Lower threshold for terrain matching (more variable than icons)

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
        threshold: Optional[float] = None,
        use_calibrated_region: bool = False
    ) -> Optional[Dict]:
        """Detect Shifting Earth event on the map.

        Args:
            map_region: Extracted map region image.
            threshold: Detection confidence threshold.
            use_calibrated_region: Deprecated, ignored. Always searches entire map.

        Returns:
            Detection result with event type and confidence, or None.
        """
        if not self.templates_loaded:
            logger.warning("Templates not loaded. Call load_templates() first.")
            return None

        threshold = threshold if threshold is not None else self.match_threshold

        best_match = None
        best_confidence = 0

        # Search the entire map for all shifting earth templates
        for event_name in SHIFTING_EARTH_MAPPING.keys():
            template_name = f"shifting_earth/{event_name}"

            if template_name not in self.matcher.templates:
                continue

            # Try template matching across the entire map
            result = self.matcher.match_template_multi_scale(
                map_region,
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
                        "scale": scale
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

        results = []

        # Search entire map for each template
        for event_name in SHIFTING_EARTH_MAPPING.keys():
            template_name = f"shifting_earth/{event_name}"

            if template_name not in self.matcher.templates:
                results.append({
                    "event": SHIFTING_EARTH_MAPPING.get(event_name, event_name),
                    "event_name": event_name,
                    "confidence": 0,
                    "error": "Template not loaded"
                })
                continue

            result = self.matcher.match_template_multi_scale(
                map_region,
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
                    "match_position": {"x": x, "y": y}
                })

        # Sort by confidence
        results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return results
