"""Field Boss detection using OCR to read boss names from screen."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Load boss data from JSON file
_BOSSES_JSON_PATH = Path(__file__).parent.parent.parent / "data" / "bosses.json"


def _load_boss_data():
    """Load boss data from bosses.json file."""
    try:
        with open(_BOSSES_JSON_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Boss data file not found: {_BOSSES_JSON_PATH}")
        return {"nightlords": {}, "field_bosses": {}}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in boss data file: {e}")
        return {"nightlords": {}, "field_bosses": {}}


def _build_field_boss_data(bosses_json: dict) -> dict:
    """Build FIELD_BOSS_DATA dict from bosses.json structure."""
    field_boss_data = {}
    for boss_key, boss_info in bosses_json.get("field_bosses", {}).items():
        boss_entry = {
            "negations": boss_info["negations"],
            "status_resistances": boss_info.get("status_resistances", {})
        }
        field_boss_data[boss_key] = boss_entry
        # Also add alternate names as keys pointing to the same data
        for alt_name in boss_info.get("names", []):
            if alt_name != boss_key:
                field_boss_data[alt_name] = boss_entry
    return field_boss_data


def _build_ocr_corrections(bosses_json: dict) -> dict:
    """Build OCR_CORRECTIONS dict from bosses.json structure."""
    corrections = {}
    for boss_key, boss_info in bosses_json.get("field_bosses", {}).items():
        canonical_name = boss_info.get("names", [boss_key])[0] if boss_info.get("names") else boss_key
        for ocr_spelling in boss_info.get("ocr_spellings", []):
            corrections[ocr_spelling] = canonical_name
    return corrections


# Load and build the data structures
_BOSSES_JSON = _load_boss_data()
FIELD_BOSS_DATA = _build_field_boss_data(_BOSSES_JSON)
OCR_CORRECTIONS = _build_ocr_corrections(_BOSSES_JSON)


def get_all_boss_data() -> dict:
    """Get the complete boss data from bosses.json."""
    return _BOSSES_JSON


def reload_boss_data():
    """Reload boss data from disk (useful for development)."""
    global _BOSSES_JSON, FIELD_BOSS_DATA, OCR_CORRECTIONS
    _BOSSES_JSON = _load_boss_data()
    FIELD_BOSS_DATA = _build_field_boss_data(_BOSSES_JSON)
    OCR_CORRECTIONS = _build_ocr_corrections(_BOSSES_JSON)


class FieldBossDetector:
    """Detects field boss names using OCR."""

    def __init__(self):
        """Initialize the field boss detector."""
        self.last_detected_boss = None
        self.confidence_threshold = 30  # Minimum OCR confidence (lowered for game fonts)

    def preprocess_image(self, image: np.ndarray) -> list[np.ndarray]:
        """Preprocess image for better OCR results.

        Returns multiple preprocessed versions to try different approaches.

        Args:
            image: BGR image from OpenCV

        Returns:
            List of preprocessed images to try
        """
        results = []

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Scale up first for better OCR
        scale_factor = 3
        gray_scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Method 1: Simple threshold (for light text on dark bg)
        _, thresh1 = cv2.threshold(gray_scaled, 200, 255, cv2.THRESH_BINARY)
        results.append(thresh1)

        # Method 2: Inverted simple threshold
        _, thresh2 = cv2.threshold(gray_scaled, 200, 255, cv2.THRESH_BINARY_INV)
        results.append(thresh2)

        # Method 3: Otsu's threshold
        _, thresh3 = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(thresh3)

        # Method 4: Adaptive threshold
        thresh4 = cv2.adaptiveThreshold(
            gray_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        results.append(thresh4)

        # Method 5: Just scaled grayscale with contrast boost
        contrast = cv2.convertScaleAbs(gray_scaled, alpha=2.0, beta=0)
        results.append(contrast)

        # Method 6: For white/light text - extract bright pixels
        _, bright = cv2.threshold(gray_scaled, 180, 255, cv2.THRESH_BINARY)
        # Invert so text is dark on light (what tesseract prefers)
        bright_inv = cv2.bitwise_not(bright)
        results.append(bright_inv)

        return results

    def extract_text(self, image: np.ndarray) -> Tuple[str, int]:
        """Extract text from image using OCR.

        Tries multiple preprocessing methods and returns best result.

        Args:
            image: Image region containing text

        Returns:
            Tuple of (extracted text, confidence score)
        """
        # Get multiple preprocessed versions
        processed_images = self.preprocess_image(image)

        best_text = ""
        best_confidence = 0

        for idx, processed in enumerate(processed_images):
            try:
                # Save debug images
                cv2.imwrite(f"debug_ocr_method_{idx}.jpg", processed)

                # Try with different PSM modes
                # PSM 7 = single line, PSM 6 = single block
                for psm in [7, 6, 3]:
                    config = f'--psm {psm}'
                    data = pytesseract.image_to_data(
                        processed,
                        output_type=pytesseract.Output.DICT,
                        config=config
                    )

                    # Combine words with decent confidence
                    words = []
                    confidences = []

                    for i, conf in enumerate(data['conf']):
                        try:
                            conf_int = int(conf)
                            if conf_int > 20:  # Lower threshold
                                word = data['text'][i].strip()
                                if word and len(word) > 1:  # Skip single chars
                                    words.append(word)
                                    confidences.append(conf_int)
                        except (ValueError, TypeError):
                            continue

                    text = ' '.join(words)
                    avg_confidence = int(np.mean(confidences)) if confidences else 0

                    logger.debug(f"OCR method {idx} psm {psm}: '{text}' (conf: {avg_confidence})")

                    if avg_confidence > best_confidence and len(text) > 3:
                        best_text = text
                        best_confidence = avg_confidence

            except Exception as e:
                logger.error(f"OCR error on method {idx}: {e}")
                continue

        return best_text, best_confidence

    def correct_ocr_text(self, text: str) -> str:
        """Apply known corrections for OCR misreadings.

        Args:
            text: Raw OCR text

        Returns:
            Corrected text
        """
        # Check direct corrections
        if text in OCR_CORRECTIONS:
            return OCR_CORRECTIONS[text]

        # Try fuzzy matching against known bosses
        text_lower = text.lower()
        for boss_name in FIELD_BOSS_DATA.keys():
            if boss_name.lower() in text_lower or text_lower in boss_name.lower():
                return boss_name

        return text

    def find_matching_boss(self, text: str) -> Optional[str]:
        """Find the best matching boss name from extracted text.

        Args:
            text: OCR extracted text

        Returns:
            Matched boss name or None
        """
        if not text:
            return None

        # Apply corrections
        corrected = self.correct_ocr_text(text)

        # Direct match
        if corrected in FIELD_BOSS_DATA:
            return corrected

        # Fuzzy match - check if any boss name is contained in the text
        text_lower = text.lower()
        for boss_name in FIELD_BOSS_DATA.keys():
            boss_lower = boss_name.lower()
            # Check for significant overlap
            if boss_lower in text_lower:
                return boss_name
            # Check individual words
            boss_words = boss_lower.split()
            matches = sum(1 for word in boss_words if word in text_lower)
            if matches >= len(boss_words) - 1 and matches > 0:
                return boss_name

        return None

    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """Detect field boss from image region.

        Args:
            image: Image region containing boss name text

        Returns:
            Detection result with boss name and weaknesses, or None
        """
        # Extract text
        raw_text, confidence = self.extract_text(image)

        logger.info(f"OCR raw text: '{raw_text}' (confidence: {confidence})")

        if confidence < self.confidence_threshold:
            logger.warning(f"OCR confidence too low: {confidence}")
            return None

        # Find matching boss
        boss_name = self.find_matching_boss(raw_text)

        if not boss_name:
            logger.info(f"No matching boss found for text: '{raw_text}'")
            return None

        # Get boss data
        boss_data = FIELD_BOSS_DATA.get(boss_name)

        if not boss_data:
            return None

        logger.info(f"Field boss detected: {boss_name}")

        return {
            "boss_name": boss_name,
            "raw_text": raw_text,
            "confidence": confidence,
            "negations": boss_data["negations"],
            "status_resistances": boss_data.get("status_resistances", {})
        }

    @staticmethod
    def get_boss_data(boss_name: str) -> Optional[Dict]:
        """Get weakness data for a boss by name.

        Args:
            boss_name: Name of the boss

        Returns:
            Boss data dict or None
        """
        return FIELD_BOSS_DATA.get(boss_name)

    @staticmethod
    def list_known_bosses() -> list:
        """Get list of all known field bosses."""
        return list(FIELD_BOSS_DATA.keys())
