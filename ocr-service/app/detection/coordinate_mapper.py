"""Coordinate mapping from pixel positions to slot IDs."""

import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

from ..config import settings, BUILDING_SLOT_COORDINATES, NIGHTLORD_COORDINATE

logger = logging.getLogger(__name__)


class CoordinateMapper:
    """Maps pixel coordinates to building slot IDs."""

    def __init__(self):
        """Initialize coordinate mapper."""
        self.slot_coordinates = BUILDING_SLOT_COORDINATES
        self.nightlord_coordinate = NIGHTLORD_COORDINATE
        self.system_size = settings.coordinate_system_size
        self.distance_threshold = settings.slot_match_distance_threshold
        # Offset to align OCR region with frontend coordinate system
        self.offset_x = settings.coordinate_offset_x
        self.offset_y = settings.coordinate_offset_y

    def pixel_to_system(
        self,
        pixel_x: int,
        pixel_y: int,
        region_width: int,
        region_height: int
    ) -> Tuple[int, int]:
        """Convert pixel coordinates to 1000x1000 coordinate system.

        Args:
            pixel_x: X coordinate in pixels.
            pixel_y: Y coordinate in pixels.
            region_width: Width of the map region in pixels.
            region_height: Height of the map region in pixels.

        Returns:
            Tuple of (x, y) in 1000x1000 system.
        """
        scale_x = self.system_size / region_width
        scale_y = self.system_size / region_height

        # Apply offset to align with frontend coordinate system
        system_x = int(pixel_x * scale_x) - self.offset_x
        system_y = int(pixel_y * scale_y) - self.offset_y

        return system_x, system_y

    def find_nearest_slot(
        self,
        system_x: int,
        system_y: int,
        threshold: Optional[int] = None
    ) -> Optional[str]:
        """Find the nearest building slot to a coordinate.

        Args:
            system_x: X coordinate in 1000x1000 system.
            system_y: Y coordinate in 1000x1000 system.
            threshold: Maximum distance to consider a match.

        Returns:
            Slot ID string (e.g., "15") or None if no slot nearby.
        """
        threshold = threshold or self.distance_threshold

        min_distance = float('inf')
        nearest_slot = None

        for slot in self.slot_coordinates:
            distance = self._euclidean_distance(
                system_x, system_y,
                slot["x"], slot["y"]
            )

            if distance < min_distance:
                min_distance = distance
                nearest_slot = slot["id"]

        if min_distance <= threshold:
            return nearest_slot

        return None

    def map_detection_to_slot(
        self,
        detection: Dict,
        region_width: int,
        region_height: int,
        threshold: Optional[int] = None
    ) -> Optional[Dict]:
        """Map a detection result to a slot ID.

        Args:
            detection: Detection result with position.
            region_width: Width of the map region.
            region_height: Height of the map region.
            threshold: Maximum distance threshold.

        Returns:
            Updated detection with slot_id, or None if no match.
        """
        if "position" not in detection:
            return None

        pixel_x = detection["position"]["x"]
        pixel_y = detection["position"]["y"]

        system_x, system_y = self.pixel_to_system(
            pixel_x, pixel_y, region_width, region_height
        )

        slot_id = self.find_nearest_slot(system_x, system_y, threshold)

        if slot_id:
            return {
                **detection,
                "slot_id": slot_id,
                "system_coords": {"x": system_x, "y": system_y}
            }

        return None

    def map_detections_to_slots(
        self,
        detections: List[Dict],
        region_width: int,
        region_height: int,
        threshold: Optional[int] = None
    ) -> List[Dict]:
        """Map multiple detections to slot IDs.

        Args:
            detections: List of detection results.
            region_width: Width of the map region.
            region_height: Height of the map region.
            threshold: Maximum distance threshold.

        Returns:
            List of detections with slot_id added.
        """
        mapped = []

        for detection in detections:
            result = self.map_detection_to_slot(
                detection, region_width, region_height, threshold
            )
            if result:
                mapped.append(result)

        return mapped

    def get_slot_coordinate(self, slot_id: str) -> Optional[Dict]:
        """Get the coordinate for a slot ID.

        Args:
            slot_id: Slot ID string.

        Returns:
            Coordinate dictionary or None.
        """
        for slot in self.slot_coordinates:
            if slot["id"] == slot_id:
                return slot
        return None

    def coordinate_to_pixel(
        self,
        system_x: int,
        system_y: int,
        region_width: int,
        region_height: int
    ) -> Tuple[int, int]:
        """Convert 1000x1000 coordinates to pixel coordinates.

        Args:
            system_x: X coordinate in 1000x1000 system.
            system_y: Y coordinate in 1000x1000 system.
            region_width: Width of the map region.
            region_height: Height of the map region.

        Returns:
            Tuple of (pixel_x, pixel_y).
        """
        scale_x = region_width / self.system_size
        scale_y = region_height / self.system_size

        pixel_x = int(system_x * scale_x)
        pixel_y = int(system_y * scale_y)

        return pixel_x, pixel_y

    def _euclidean_distance(
        self,
        x1: int, y1: int,
        x2: int, y2: int
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_all_slots_within_distance(
        self,
        system_x: int,
        system_y: int,
        max_distance: int
    ) -> List[Dict]:
        """Find all slots within a distance from a point.

        Args:
            system_x: X coordinate in 1000x1000 system.
            system_y: Y coordinate in 1000x1000 system.
            max_distance: Maximum distance.

        Returns:
            List of slots with distances.
        """
        results = []

        for slot in self.slot_coordinates:
            distance = self._euclidean_distance(
                system_x, system_y,
                slot["x"], slot["y"]
            )

            if distance <= max_distance:
                results.append({
                    "slot_id": slot["id"],
                    "distance": distance,
                    "x": slot["x"],
                    "y": slot["y"]
                })

        results.sort(key=lambda s: s["distance"])
        return results
