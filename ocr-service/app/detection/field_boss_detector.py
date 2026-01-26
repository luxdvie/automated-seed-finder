"""Field Boss detection using OCR to read boss names from screen."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)

# Field boss names mapped to their weakness data (from Fextralife wiki)
# Negative = weak to (takes more damage), Positive = resistant to (takes less damage)
FIELD_BOSS_DATA = {
    # From eip.gg field boss list + Fextralife data
    "Ancestor Spirit": {
        "negations": {"standard": 0, "slash": -10, "strike": 0, "pierce": 0, "magic": 20, "fire": -20, "lightning": 0, "holy": -20}
    },
    "Ancient Hero of Zamor": {
        "negations": {"standard": 10, "slash": 10, "strike": 10, "pierce": 0, "magic": 0, "fire": -20, "lightning": -20, "holy": 0}
    },
    "Bell Bearing Hunter": {
        "negations": {"standard": 35, "slash": 35, "strike": 35, "pierce": -10, "magic": 40, "fire": 40, "lightning": 20, "holy": 40}
    },
    "Black Knife Assassin": {
        "negations": {"standard": 10, "slash": 10, "strike": 10, "pierce": 35, "magic": 20, "fire": 20, "lightning": 20, "holy": 40}
    },
    "Death Rite Bird": {
        "negations": {"standard": 10, "slash": 10, "strike": -40, "pierce": 35, "magic": 20, "fire": 20, "lightning": 40, "holy": -40}
    },
    "Demi-Human Queen": {
        "negations": {"standard": 0, "slash": -10, "strike": 0, "pierce": 0, "magic": 0, "fire": -20, "lightning": 0, "holy": 0}
    },
    "Draconic Tree Sentinel": {
        "negations": {"standard": 10, "slash": 35, "strike": 10, "pierce": 10, "magic": 20, "fire": 40, "lightning": 40, "holy": 20}
    },
    "Elder Lion": {
        "negations": {"standard": 0, "slash": -10, "strike": 0, "pierce": 0, "magic": 0, "fire": -20, "lightning": 0, "holy": 0}
    },
    "Erdtree Avatar": {
        "negations": {"standard": 10, "slash": 10, "strike": 0, "pierce": 10, "magic": 20, "fire": -40, "lightning": 20, "holy": 40}
    },
    "Flying Dragon": {
        "negations": {"standard": 35, "slash": 35, "strike": 35, "pierce": 10, "magic": 40, "fire": 40, "lightning": 40, "holy": 40}
    },
    "Flying Dragon of the Hills": {
        "negations": {"standard": 35, "slash": 35, "strike": 35, "pierce": 10, "magic": 40, "fire": 40, "lightning": 40, "holy": 40}
    },
    "Golden Hippopotamus": {
        "negations": {"standard": 0, "slash": -10, "strike": 0, "pierce": 0, "magic": 0, "fire": -20, "lightning": -20, "holy": 0}
    },
    "Leonine Misbegotten": {
        "negations": {"standard": 10, "slash": 0, "strike": 10, "pierce": 10, "magic": 20, "fire": 0, "lightning": 20, "holy": 20}
    },
    "Miranda Blossom": {
        "negations": {"standard": -10, "slash": -40, "strike": 10, "pierce": -10, "magic": -20, "fire": -40, "lightning": 20, "holy": 20}
    },
    "Night's Cavalry": {
        "negations": {"standard": 35, "slash": 35, "strike": 35, "pierce": 10, "magic": 40, "fire": 40, "lightning": 20, "holy": 40}
    },
    "Red Wolf of the King Consort": {
        "negations": {"standard": 0, "slash": -10, "strike": 0, "pierce": 0, "magic": 40, "fire": 20, "lightning": 20, "holy": 20}
    },
    "Royal Carian Knight": {
        "negations": {"standard": 10, "slash": 10, "strike": 10, "pierce": 10, "magic": 40, "fire": 40, "lightning": 0, "holy": 20}
    },
    "Royal Revenant": {
        "negations": {"standard": 0, "slash": -10, "strike": 0, "pierce": 0, "magic": 0, "fire": 0, "lightning": 0, "holy": 40}
    },
    "Tree Sentinel": {
        "negations": {"standard": 10, "slash": 35, "strike": 10, "pierce": 10, "magic": 20, "fire": 40, "lightning": 0, "holy": 40}
    },
    "Ulcerated Tree Spirit": {
        "negations": {"standard": 0, "slash": 0, "strike": 0, "pierce": 0, "magic": 20, "fire": -20, "lightning": 20, "holy": 40}
    },
    # Evergaol Bosses
    "Ancient Dragon": {
        "negations": {"standard": 35, "slash": 35, "strike": 35, "pierce": 10, "magic": 40, "fire": 40, "lightning": 80, "holy": 40}
    },
    "Banished Knights": {
        "negations": {"standard": 10, "slash": 35, "strike": 10, "pierce": 0, "magic": 20, "fire": 20, "lightning": -20, "holy": 20}
    },
    "Beastmen of Farum Azula": {
        "negations": {"standard": 0, "slash": 0, "strike": -10, "pierce": 0, "magic": 0, "fire": -20, "lightning": 20, "holy": 0}
    },
    "Bloodhound Knight": {
        "negations": {"standard": 10, "slash": 10, "strike": 10, "pierce": 0, "magic": 0, "fire": 0, "lightning": -20, "holy": 0}
    },
    "Crucible Knight": {
        "negations": {"standard": 35, "slash": 35, "strike": 35, "pierce": 35, "magic": 40, "fire": 20, "lightning": 20, "holy": 40}
    },
    "Crystalians": {
        "negations": {"standard": 10, "slash": 35, "strike": 10, "pierce": 35, "magic": 40, "fire": 40, "lightning": 40, "holy": 40}
    },
    "Dragonkin Soldier": {
        "negations": {"standard": 10, "slash": 0, "strike": 10, "pierce": 10, "magic": 20, "fire": 20, "lightning": 40, "holy": 20}
    },
    "Godskin Apostle": {
        "negations": {"standard": 0, "slash": -10, "strike": 10, "pierce": 0, "magic": 20, "fire": 40, "lightning": 20, "holy": 40}
    },
    "Godskin Noble": {
        "negations": {"standard": 0, "slash": -10, "strike": 35, "pierce": 0, "magic": 20, "fire": 40, "lightning": 20, "holy": 40}
    },
    "Godskin Noble & Apostle": {
        "negations": {"standard": 0, "slash": -10, "strike": 20, "pierce": 0, "magic": 20, "fire": 40, "lightning": 20, "holy": 40}
    },
    "Grave Warden Duelist": {
        "negations": {"standard": 0, "slash": -10, "strike": 0, "pierce": 0, "magic": 0, "fire": 0, "lightning": 0, "holy": 0}
    },
    "Nox Warriors": {
        "negations": {"standard": 0, "slash": 0, "strike": 0, "pierce": 0, "magic": 0, "fire": 0, "lightning": 0, "holy": 0}
    },
    "Omen": {
        "negations": {"standard": 0, "slash": -10, "strike": 0, "pierce": 0, "magic": 0, "fire": 0, "lightning": 0, "holy": 20}
    },
    "Beastly Brigade": {
        "negations": {"standard": 0, "slash": 0, "strike": -10, "pierce": 0, "magic": 0, "fire": -20, "lightning": 20, "holy": 0}
    },
    "Stoneskin Lords": {
        "negations": {"standard": 10, "slash": 35, "strike": 10, "pierce": 35, "magic": 40, "fire": 40, "lightning": 40, "holy": 40}
    },
    # DLC Bosses
    "Blackgaol Knight": {
        "negations": {"standard": 35, "slash": 35, "strike": 35, "pierce": 35, "magic": 25, "fire": 30, "lightning": 25, "holy": 25}
    },
    "Valiant Gargoyle": {
        "negations": {"standard": 10, "slash": 35, "strike": 0, "pierce": 35, "magic": 20, "fire": 40, "lightning": 40, "holy": 40}
    },
    "Fallingstar Beast": {
        "negations": {"standard": 35, "slash": 35, "strike": 35, "pierce": 35, "magic": 20, "fire": 20, "lightning": 20, "holy": 20}
    },
}

# Common OCR misreadings to correct
OCR_CORRECTIONS = {
    # Demi-Human Queen variations
    "Demi-Human Oueen": "Demi-Human Queen",
    "Demi Human Queen": "Demi-Human Queen",
    "DemiHuman Queen": "Demi-Human Queen",
    "Demi-Human Gueen": "Demi-Human Queen",
    # Tree Sentinel variations
    "Tree Sent1nel": "Tree Sentinel",
    "Tree Sentine1": "Tree Sentinel",
    # Night's Cavalry variations
    "N1ght's Cavalry": "Night's Cavalry",
    "Night's Cava1ry": "Night's Cavalry",
    "Nights Cavalry": "Night's Cavalry",
    # Flying Dragon variations
    "F1ying Dragon": "Flying Dragon",
    "Flying Oragon": "Flying Dragon",
    # Erdtree Avatar variations
    "Erdtree Avater": "Erdtree Avatar",
    # Death Rite Bird variations
    "Death R1te Bird": "Death Rite Bird",
    "Death Rite 8ird": "Death Rite Bird",
    # Leonine Misbegotten variations
    "Leon1ne Misbegotten": "Leonine Misbegotten",
    "Leonine Misbegoten": "Leonine Misbegotten",
    # Miranda Blossom variations
    "M1randa Blossom": "Miranda Blossom",
    # Royal variations
    "Roya1 Revenant": "Royal Revenant",
    "Royal Carian Kn1ght": "Royal Carian Knight",
    "Roya1 Carian Knight": "Royal Carian Knight",
    # Bell Bearing Hunter
    "Be11 Bearing Hunter": "Bell Bearing Hunter",
    "Bell 8earing Hunter": "Bell Bearing Hunter",
    # Ulcerated Tree Spirit
    "U1cerated Tree Spirit": "Ulcerated Tree Spirit",
    "Ulcerated Tree Sp1rit": "Ulcerated Tree Spirit",
    # Golden Hippopotamus
    "Golden Hippopotarnus": "Golden Hippopotamus",
    "Go1den Hippopotamus": "Golden Hippopotamus",
    # Elder Lion
    "E1der Lion": "Elder Lion",
    "Elder L1on": "Elder Lion",
    # Ancient Hero of Zamor
    "Anc1ent Hero of Zamor": "Ancient Hero of Zamor",
    "Ancient Hero of Zamor": "Ancient Hero of Zamor",
    # Black Knife Assassin
    "B1ack Knife Assassin": "Black Knife Assassin",
    "Black Kn1fe Assassin": "Black Knife Assassin",
    # Draconic Tree Sentinel
    "Dracon1c Tree Sentinel": "Draconic Tree Sentinel",
    "Draconic Tree Sent1nel": "Draconic Tree Sentinel",
    # Red Wolf
    "Red Wo1f of the King Consort": "Red Wolf of the King Consort",
    "Red Wolf of the K1ng Consort": "Red Wolf of the King Consort",
    # Ancestor Spirit
    "Ancestor Sp1rit": "Ancestor Spirit",
    # Evergaol Bosses
    "Anc1ent Dragon": "Ancient Dragon",
    "Ancient Oragon": "Ancient Dragon",
    "Ban1shed Knights": "Banished Knights",
    "Banished Kn1ghts": "Banished Knights",
    "Beastmen of Farum Azu1a": "Beastmen of Farum Azula",
    "Beastmen of Farurn Azula": "Beastmen of Farum Azula",
    "B1oodhound Knight": "Bloodhound Knight",
    "Bloodhound Kn1ght": "Bloodhound Knight",
    "Cruc1ble Knight": "Crucible Knight",
    "Crucible Kn1ght": "Crucible Knight",
    "Crysta1ians": "Crystalians",
    "Crystal1ans": "Crystalians",
    "Dragon1in Soldier": "Dragonkin Soldier",
    "Dragonkin So1dier": "Dragonkin Soldier",
    "Dragonk1n Soldier": "Dragonkin Soldier",
    "Godskin Apost1e": "Godskin Apostle",
    "Godsk1n Apostle": "Godskin Apostle",
    "Godskin Nob1e": "Godskin Noble",
    "Godsk1n Noble": "Godskin Noble",
    "Godskin Noble & Apost1e": "Godskin Noble & Apostle",
    "Godsk1n Noble & Apostle": "Godskin Noble & Apostle",
    "Grave Warden Due1ist": "Grave Warden Duelist",
    "Grave Warden Duel1st": "Grave Warden Duelist",
    "Nox Warr1ors": "Nox Warriors",
    "Ornon": "Omen",
    "0men": "Omen",
    "Beast1y Brigade": "Beastly Brigade",
    "Beastly Br1gade": "Beastly Brigade",
    "Stoneskin Lords": "Stoneskin Lords",
    "Stonesk1n Lords": "Stoneskin Lords",
    # DLC Bosses
    "B1ackgaol Knight": "Blackgaol Knight",
    "Blackgaol Kn1ght": "Blackgaol Knight",
    "Blackgao1 Knight": "Blackgaol Knight",
    "Va1iant Gargoyle": "Valiant Gargoyle",
    "Valiant Gargoy1e": "Valiant Gargoyle",
    "Valiant Gargoyle": "Valiant Gargoyle",
    "Fa1lingstar Beast": "Fallingstar Beast",
    "Fallingstar 8east": "Fallingstar Beast",
    "Fall1ngstar Beast": "Fallingstar Beast",
}


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
            "negations": boss_data["negations"]
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
