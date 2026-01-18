"""POI (Point of Interest) building icon detection."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

from .template_matcher import TemplateMatcher
from ..config import settings

logger = logging.getLogger(__name__)


class POIDetector:
    """Detects building icons on the in-game map."""

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize POI detector.

        Args:
            templates_dir: Directory containing template images.
        """
        self.matcher = TemplateMatcher(templates_dir)
        self.templates_loaded = False

    def load_templates(self) -> bool:
        """Load building templates.

        Returns:
            True if templates loaded successfully.
        """
        count = self.matcher.load_templates_from_directory("buildings")
        self.templates_loaded = count > 0

        if not self.templates_loaded:
            logger.warning("No building templates found. POI detection will be limited.")

        return self.templates_loaded

    def detect(
        self,
        map_region: np.ndarray,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Detect all building icons on the map.

        Args:
            map_region: Extracted map region image.
            threshold: Detection confidence threshold.

        Returns:
            List of detected buildings with positions and types.
        """
        if not self.templates_loaded:
            logger.warning("Templates not loaded. Call load_templates() first.")
            return []

        threshold = threshold or settings.template_match_threshold

        # Get all matches for building templates
        matches = self.matcher.match_all_templates(map_region, "buildings", threshold)

        # Filter duplicates (same building at similar positions)
        filtered = self._filter_duplicate_detections(matches)

        return filtered

    def detect_single(
        self,
        map_region: np.ndarray,
        building_type: str,
        threshold: Optional[float] = None
    ) -> Optional[Dict]:
        """Detect a specific building type on the map.

        Args:
            map_region: Extracted map region image.
            building_type: Type of building to detect.
            threshold: Detection confidence threshold.

        Returns:
            Detection result or None.
        """
        template_name = f"buildings/{building_type}"
        result = self.matcher.match_template_multi_scale(
            map_region, template_name, threshold=threshold
        )

        if result:
            x, y, confidence, scale = result
            return {
                "building_type": building_type,
                "position": {"x": x, "y": y},
                "confidence": confidence,
                "scale": scale
            }

        return None

    def _filter_duplicate_detections(
        self,
        detections: List[Dict],
        distance_threshold: int = 30
    ) -> List[Dict]:
        """Filter duplicate detections at similar positions.

        Args:
            detections: List of detection results.
            distance_threshold: Minimum distance between unique detections.

        Returns:
            Filtered list of detections.
        """
        if not detections:
            return []

        filtered = []

        for detection in detections:
            is_duplicate = False

            for existing in filtered:
                # Calculate distance between centers
                dx = detection["x"] - existing["position"]["x"]
                dy = detection["y"] - existing["position"]["y"]
                distance = np.sqrt(dx * dx + dy * dy)

                if distance < distance_threshold:
                    # Keep the one with higher confidence
                    if detection["confidence"] > existing["confidence"]:
                        existing["building_type"] = detection["template"]
                        existing["position"] = {"x": detection["x"], "y": detection["y"]}
                        existing["confidence"] = detection["confidence"]
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append({
                    "building_type": detection["template"],
                    "position": {"x": detection["x"], "y": detection["y"]},
                    "confidence": detection["confidence"]
                })

        return filtered

    def detect_in_region(
        self,
        map_region: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Detect buildings within a specific region.

        Args:
            map_region: Full map region image.
            x1, y1, x2, y2: Bounding box of search region.
            threshold: Detection confidence threshold.

        Returns:
            List of detected buildings within the region.
        """
        # Crop to search region
        search_region = map_region[y1:y2, x1:x2]

        if search_region.size == 0:
            return []

        # Detect in cropped region
        detections = self.detect(search_region, threshold)

        # Adjust positions to full map coordinates
        for detection in detections:
            detection["position"]["x"] += x1
            detection["position"]["y"] += y1

        return detections
