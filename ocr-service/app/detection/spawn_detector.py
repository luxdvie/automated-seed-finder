"""Player spawn position detection using template matching."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class SpawnDetector:
    """Detects player spawn marker using template matching."""

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize spawn detector."""
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent.parent.parent / "templates"
        self.spawn_templates: Dict[str, np.ndarray] = {}
        self.templates_loaded = False

        # Template matching threshold
        self.match_threshold = 0.65

        # Position constraints (in ratio of image dimensions)
        self.min_x_ratio = 0.15  # Left 15% might have nightlord/event icons
        self.max_x_ratio = 0.90  # Right 10% might have UI
        self.min_y_ratio = 0.10  # Top 10% margin
        self.max_y_ratio = 0.95  # Bottom 5% might have UI

        # Fallback: HSV color detection (if no templates)
        # Red wraps around in HSV, so we need two ranges
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([15, 255, 255])
        self.red_lower2 = np.array([165, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])

        # Green range
        self.green_lower = np.array([30, 50, 50])
        self.green_upper = np.array([90, 255, 255])

        # Blue range (broad to catch various shades)
        self.blue_lower = np.array([85, 50, 50])
        self.blue_upper = np.array([135, 255, 255])

        # Alternative lighter blue range (kept for compatibility)
        self.light_blue_lower = np.array([85, 30, 100])
        self.light_blue_upper = np.array([135, 255, 255])

        # Player badge size constraints (~68x68 pixels)
        self.min_contour_area = 2000
        self.max_contour_area = 6000
        self.min_badge_diameter = 50
        self.max_badge_diameter = 90

    def load_templates(self) -> bool:
        """Load spawn badge templates.

        Returns:
            True if at least one template loaded.
        """
        spawn_dir = self.templates_dir / "spawn"
        if not spawn_dir.exists():
            logger.warning(f"Spawn templates directory not found: {spawn_dir}")
            return False

        count = 0
        for file in spawn_dir.glob("*.png"):
            template = cv2.imread(str(file))
            if template is not None:
                name = file.stem  # e.g., "blue", "red", "green"
                self.spawn_templates[name] = template
                logger.info(f"Loaded spawn template: {name} ({template.shape[1]}x{template.shape[0]})")
                count += 1

        self.templates_loaded = count > 0
        logger.info(f"Loaded {count} spawn templates")
        return self.templates_loaded

    def detect_with_template(
        self,
        map_region: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[Dict]:
        """Detect spawn using template matching.

        Args:
            map_region: Map region image.
            threshold: Match threshold (0-1).

        Returns:
            Detection result or None.
        """
        if not self.spawn_templates:
            return None

        threshold = threshold if threshold is not None else self.match_threshold
        height, width = map_region.shape[:2]

        best_match = None
        best_confidence = 0

        for name, template in self.spawn_templates.items():
            # Try multiple scales
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                new_w = int(template.shape[1] * scale)
                new_h = int(template.shape[0] * scale)

                if new_w < 20 or new_h < 20:
                    continue
                if new_w > width or new_h > height:
                    continue

                scaled = cv2.resize(template, (new_w, new_h))

                # Match template
                result = cv2.matchTemplate(map_region, scaled, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > best_confidence:
                    # Check position constraints
                    center_x = max_loc[0] + new_w // 2
                    center_y = max_loc[1] + new_h // 2

                    x_ratio = center_x / width
                    y_ratio = center_y / height

                    if (self.min_x_ratio <= x_ratio <= self.max_x_ratio and
                        self.min_y_ratio <= y_ratio <= self.max_y_ratio):
                        best_confidence = max_val
                        best_match = {
                            "position": {"x": center_x, "y": center_y},
                            "confidence": max_val,
                            "template": name,
                            "scale": scale
                        }

        if best_match and best_match["confidence"] >= threshold:
            return best_match

        return None

    def detect(
        self,
        map_region: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[Dict]:
        """Detect player spawn marker on the map.

        Uses Hough Circle detection to find the distinctive double-ring pattern
        of the player badge (outer circle + inner circle).

        Args:
            map_region: Extracted map region image.
            threshold: Detection confidence threshold.

        Returns:
            Detection result with position and confidence, or None.
        """
        if map_region is None or map_region.size == 0:
            return None

        # Try template matching first if templates are loaded
        if self.templates_loaded:
            result = self.detect_with_template(map_region, threshold)
            if result:
                return result

        # Fall back to Hough Circle detection for double-ring pattern
        try:
            result = self._detect_with_hough_circles(map_region)
            if result:
                return result
        except Exception as e:
            logger.error(f"Error in Hough circle detection: {e}")

        # Last resort: color-based detection
        try:
            return self._detect_with_color(map_region)
        except Exception as e:
            logger.error(f"Error detecting spawn marker: {e}")
            return None

    def _check_player_color(self, map_region: np.ndarray, cx: int, cy: int, radius: int) -> Tuple[bool, str]:
        """Check if the region has player badge colors (red, green, or blue).

        Returns:
            Tuple of (is_player_color, detected_color)
        """
        height, width = map_region.shape[:2]

        # Extract region around the circle
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(width, cx + radius)
        y2 = min(height, cy + radius)

        region = map_region[y1:y2, x1:x2]
        if region.size == 0:
            return False, "none"

        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Check for player colors (red, green, blue) with decent saturation
        # Red (wraps around in HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
        red_count = np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)

        # Green
        green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
        green_count = np.sum(green_mask > 0)

        # Blue
        blue_mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([130, 255, 255]))
        blue_count = np.sum(blue_mask > 0)

        total_pixels = region.shape[0] * region.shape[1]
        min_color_ratio = 0.15  # At least 15% of pixels should be colored

        # Find dominant color
        max_count = max(red_count, green_count, blue_count)
        color_ratio = max_count / total_pixels

        if color_ratio < min_color_ratio:
            return False, "gray"  # Not enough color - likely a castle/stone icon

        if max_count == red_count:
            return True, "red"
        elif max_count == green_count:
            return True, "green"
        else:
            return True, "blue"

    def _detect_with_hough_circles(self, map_region: np.ndarray) -> Optional[Dict]:
        """Detect spawn using Hough Circle detection for double-ring pattern."""
        height, width = map_region.shape[:2]
        gray = cv2.cvtColor(map_region, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles - looking for the outer ring (~34 radius) and inner ring (~20 radius)
        # Player badge is ~68px diameter, so outer radius ~34, inner radius ~20
        outer_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=25,
            maxRadius=45
        )

        if outer_circles is None:
            return None

        candidates = []
        outer_circles = np.uint16(np.around(outer_circles))

        for circle in outer_circles[0, :]:
            cx, cy, radius = circle[0], circle[1], circle[2]

            # Check position constraints
            x_ratio = cx / width
            y_ratio = cy / height
            if not (self.min_x_ratio <= x_ratio <= self.max_x_ratio):
                continue
            if not (self.min_y_ratio <= y_ratio <= self.max_y_ratio):
                continue

            # Check for player badge colors (reject gray/stone castle icons)
            is_colored, color = self._check_player_color(map_region, cx, cy, radius)
            if not is_colored:
                continue  # Skip gray/stone colored circles (likely castles)

            # Look for an inner circle near this outer circle
            inner_region = blurred[max(0, cy-radius):min(height, cy+radius),
                                   max(0, cx-radius):min(width, cx+radius)]

            if inner_region.size == 0:
                continue

            inner_circles = cv2.HoughCircles(
                inner_region,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=10,
                param1=50,
                param2=25,
                minRadius=10,
                maxRadius=25
            )

            # If we found an inner circle, this is likely a player badge
            has_inner = inner_circles is not None and len(inner_circles[0]) > 0

            # Score based on whether inner circle found
            confidence = 0.85 if has_inner else 0.5

            candidates.append({
                "position": {"x": int(cx), "y": int(cy)},
                "confidence": confidence,
                "radius": int(radius),
                "has_inner_circle": has_inner,
                "color": color
            })

        if not candidates:
            return None

        # Return best candidate (prefer ones with inner circle)
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates[0]

    def _detect_with_color(self, map_region: np.ndarray) -> Optional[Dict]:
        """Fallback: detect spawn using color-based detection."""
        hsv = cv2.cvtColor(map_region, cv2.COLOR_BGR2HSV)

        # Create masks for all player marker colors (red, green, blue)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        blue_mask1 = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        blue_mask2 = cv2.inRange(hsv, self.light_blue_lower, self.light_blue_upper)
        blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)

        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        best_candidate = self._find_best_marker_candidate(contours, map_region.shape, map_region)
        return best_candidate

    def _check_edge_strength(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> float:
        """Check for strong internal edges (player badge has inner circle).

        Returns edge strength score 0-1, higher = more internal edges.
        """
        # Extract region with some padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Count edge pixels in the inner region (not the outer edge)
        inner_margin = 10
        if edges.shape[0] > inner_margin * 2 and edges.shape[1] > inner_margin * 2:
            inner_edges = edges[inner_margin:-inner_margin, inner_margin:-inner_margin]
            edge_density = np.sum(inner_edges > 0) / inner_edges.size
        else:
            edge_density = np.sum(edges > 0) / edges.size

        # Player badge should have edge density > 0.05 (inner circle)
        # Event fog will have very low edge density (blurred)
        return min(edge_density * 10, 1.0)  # Scale up and cap at 1.0

    def _find_best_marker_candidate(
        self,
        contours: List,
        image_shape: Tuple[int, int, int],
        original_image: np.ndarray = None
    ) -> Optional[Dict]:
        """Find the best spawn marker candidate from contours.

        Args:
            contours: List of contours from mask.
            image_shape: Shape of the image.
            original_image: Original BGR image for edge detection.

        Returns:
            Best candidate detection or None.
        """
        height, width = image_shape[:2]
        candidates = []

        # Define valid map area (exclude edges which may have UI)
        # In 1000x1000 system, valid building slots are roughly:
        # x: 199-804, y: 180-795
        # Add some margin for player movement
        min_x_ratio = 0.15  # Left 15% might have nightlord/event icons
        max_x_ratio = 0.90  # Right 10% might have UI
        min_y_ratio = 0.10  # Top 10% margin
        max_y_ratio = 0.95  # Bottom 5% might have UI

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            # Get bounding rect and center
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Filter by bounding box size (player badge ~68x68)
            if w < self.min_badge_diameter or w > self.max_badge_diameter:
                continue
            if h < self.min_badge_diameter or h > self.max_badge_diameter:
                continue

            # Filter by position - must be in valid map area
            x_ratio = center_x / width
            y_ratio = center_y / height
            if x_ratio < min_x_ratio or x_ratio > max_x_ratio:
                continue
            if y_ratio < min_y_ratio or y_ratio > max_y_ratio:
                continue

            # Calculate aspect ratio (markers are roughly circular)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.7 or aspect_ratio > 1.4:  # Stricter aspect ratio
                continue

            # Calculate circularity (require higher circularity)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            if circularity < 0.5:  # Must be fairly circular
                continue

            # Check edge strength (inner circle detection)
            edge_score = 0.0
            if original_image is not None:
                edge_score = self._check_edge_strength(original_image, x, y, w, h)
                # Require minimum edge strength to filter out blurry fog
                if edge_score < 0.15:
                    continue

            # Score based on circularity, size, and edge strength
            ideal_area = 3630  # π * 34²
            size_score = 1.0 - abs(area - ideal_area) / ideal_area
            size_score = max(0, size_score)

            # Weight: 40% circularity, 30% size, 30% edge strength
            score = circularity * 0.4 + size_score * 0.3 + edge_score * 0.3

            candidates.append({
                "x": center_x,
                "y": center_y,
                "area": area,
                "width": w,
                "height": h,
                "circularity": circularity,
                "edge_score": edge_score,
                "score": score,
                "confidence": min(score, 1.0)
            })

        if not candidates:
            return None

        # Return best candidate by score
        best = max(candidates, key=lambda c: c["score"])

        return {
            "position": {"x": best["x"], "y": best["y"]},
            "confidence": best["confidence"],
            "area": best["area"]
        }

    def detect_multiple(
        self,
        map_region: np.ndarray,
        max_count: int = 5
    ) -> List[Dict]:
        """Detect multiple potential spawn markers.

        Args:
            map_region: Extracted map region image.
            max_count: Maximum number of candidates to return.

        Returns:
            List of detection results sorted by confidence.
        """
        if map_region is None or map_region.size == 0:
            return []

        try:
            hsv = cv2.cvtColor(map_region, cv2.COLOR_BGR2HSV)

            # Red (wraps around)
            red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Green
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

            # Blue
            blue_mask1 = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
            blue_mask2 = cv2.inRange(hsv, self.light_blue_lower, self.light_blue_upper)
            blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)

            # Combine all
            combined_mask = cv2.bitwise_or(red_mask, green_mask)
            combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []
            height, width = map_region.shape[:2]

            # Same position filtering as _find_best_marker_candidate
            min_x_ratio = 0.15
            max_x_ratio = 0.90
            min_y_ratio = 0.10
            max_y_ratio = 0.95

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area or area > self.max_contour_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                # Filter by bounding box size (player badge ~68x68)
                if w < self.min_badge_diameter or w > self.max_badge_diameter:
                    continue
                if h < self.min_badge_diameter or h > self.max_badge_diameter:
                    continue

                # Filter by position
                x_ratio = center_x / width
                y_ratio = center_y / height
                if x_ratio < min_x_ratio or x_ratio > max_x_ratio:
                    continue
                if y_ratio < min_y_ratio or y_ratio > max_y_ratio:
                    continue

                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.7 or aspect_ratio > 1.4:  # Stricter
                    continue

                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                if circularity < 0.5:  # Must be fairly circular
                    continue

                # Check edge strength
                edge_score = self._check_edge_strength(map_region, x, y, w, h)
                if edge_score < 0.15:  # Require minimum edge strength
                    continue

                # Score based on circularity, size, and edge strength
                ideal_area = 3630
                size_score = 1.0 - abs(area - ideal_area) / ideal_area
                size_score = max(0, size_score)
                score = circularity * 0.4 + size_score * 0.3 + edge_score * 0.3

                candidates.append({
                    "position": {"x": center_x, "y": center_y},
                    "confidence": min(score, 1.0),
                    "area": area,
                    "width": w,
                    "height": h,
                    "edge_score": edge_score
                })

            # Sort by confidence and return top candidates
            candidates.sort(key=lambda c: c["confidence"], reverse=True)
            return candidates[:max_count]

        except Exception as e:
            logger.error(f"Error detecting spawn markers: {e}")
            return []

    def set_color_range(
        self,
        lower_hsv: Tuple[int, int, int],
        upper_hsv: Tuple[int, int, int]
    ) -> None:
        """Set custom HSV color range for detection.

        Args:
            lower_hsv: Lower HSV bounds (H, S, V).
            upper_hsv: Upper HSV bounds (H, S, V).
        """
        self.blue_lower = np.array(lower_hsv)
        self.blue_upper = np.array(upper_hsv)
        logger.info(f"Updated spawn color range: {lower_hsv} - {upper_hsv}")
