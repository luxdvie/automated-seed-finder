"""Template matching utilities for icon detection."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class TemplateMatcher:
    """Multi-scale template matching for icon detection."""

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize template matcher.

        Args:
            templates_dir: Directory containing template images.
        """
        self.templates_dir = Path(templates_dir or settings.templates_dir)
        self.templates: Dict[str, np.ndarray] = {}
        self.template_sizes: Dict[str, Tuple[int, int]] = {}

    def load_template(self, name: str, path: Path) -> bool:
        """Load a template image.

        Args:
            name: Template identifier.
            path: Path to template image file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                logger.error(f"Failed to load template: {path}")
                return False

            self.templates[name] = template
            self.template_sizes[name] = (template.shape[1], template.shape[0])  # (width, height)
            logger.info(f"Loaded template '{name}' from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading template {path}: {e}")
            return False

    def load_templates_from_directory(self, subdir: str) -> int:
        """Load all templates from a subdirectory.

        Args:
            subdir: Subdirectory name within templates_dir.

        Returns:
            Number of templates loaded.
        """
        template_path = self.templates_dir / subdir
        if not template_path.exists():
            logger.warning(f"Template directory does not exist: {template_path}")
            return 0

        count = 0
        for file in template_path.glob("*.png"):
            name = f"{subdir}/{file.stem}"
            if self.load_template(name, file):
                count += 1

        logger.info(f"Loaded {count} templates from {subdir}")
        return count

    def match_template(
        self,
        image: np.ndarray,
        template_name: str,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[int, int, float]]:
        """Match a single template against an image.

        Args:
            image: Image to search in (grayscale).
            template_name: Name of template to match.
            threshold: Confidence threshold. Defaults to config value.

        Returns:
            Tuple of (x, y, confidence) if match found, None otherwise.
        """
        threshold = threshold or settings.template_match_threshold

        if template_name not in self.templates:
            logger.warning(f"Template not loaded: {template_name}")
            return None

        template = self.templates[template_name]

        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            # Return center of match
            h, w = template.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return center_x, center_y, max_val

        return None

    def match_template_multi_scale(
        self,
        image: np.ndarray,
        template_name: str,
        scales: List[float] = None,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[int, int, float, float]]:
        """Match template at multiple scales.

        Args:
            image: Image to search in.
            template_name: Name of template to match.
            scales: List of scale factors to try.
            threshold: Confidence threshold.

        Returns:
            Tuple of (x, y, confidence, scale) if match found, None otherwise.
        """
        if scales is None:
            scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

        threshold = settings.template_match_threshold if threshold is None else threshold

        if template_name not in self.templates:
            logger.warning(f"Template not loaded: {template_name}")
            return None

        template = self.templates[template_name]

        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        best_match = None
        best_confidence = 0

        for scale in scales:
            # Resize template
            new_width = int(template.shape[1] * scale)
            new_height = int(template.shape[0] * scale)

            if new_width < 10 or new_height < 10:
                continue

            if new_width > image.shape[1] or new_height > image.shape[0]:
                continue

            scaled_template = cv2.resize(template, (new_width, new_height))

            # Match template
            result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_confidence:
                best_confidence = max_val
                center_x = max_loc[0] + new_width // 2
                center_y = max_loc[1] + new_height // 2
                best_match = (center_x, center_y, max_val, scale)

        if best_match and best_match[2] >= threshold:
            return best_match

        return None

    def match_all_templates(
        self,
        image: np.ndarray,
        category: str,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Match all templates in a category against an image.

        Args:
            image: Image to search in.
            category: Template category (e.g., "nightlords", "buildings").
            threshold: Confidence threshold.

        Returns:
            List of match results with template name and location.
        """
        threshold = threshold or settings.template_match_threshold
        matches = []

        category_prefix = f"{category}/"
        category_templates = [
            name for name in self.templates.keys()
            if name.startswith(category_prefix)
        ]

        for template_name in category_templates:
            result = self.match_template_multi_scale(image, template_name, threshold=threshold)
            if result:
                x, y, confidence, scale = result
                matches.append({
                    "template": template_name.replace(category_prefix, ""),
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "scale": scale
                })

        # Sort by confidence
        matches.sort(key=lambda m: m["confidence"], reverse=True)
        return matches

    def find_best_match(
        self,
        image: np.ndarray,
        category: str,
        threshold: Optional[float] = None
    ) -> Optional[Dict]:
        """Find the best matching template in a category.

        Args:
            image: Image to search in.
            category: Template category.
            threshold: Confidence threshold.

        Returns:
            Best match result, or None if no matches found.
        """
        matches = self.match_all_templates(image, category, threshold)
        return matches[0] if matches else None
