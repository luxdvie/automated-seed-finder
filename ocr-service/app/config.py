"""Configuration settings for OCR service."""

import json
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # Video capture settings
    capture_device_index: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 30

    # Detection thresholds
    template_match_threshold: float = 0.50
    spawn_color_threshold: float = 0.80

    # Map region extraction (percentage of screen)
    map_region_x_start: float = 0.45  # Start from 45% of screen width
    map_region_x_end: float = 1.0     # End at 100% of screen width
    map_region_y_start: float = 0.05  # Start from 5% of screen height
    map_region_y_end: float = 0.95    # End at 95% of screen height
    # Trim extra padding from left side of extracted map region (pixels)
    # The in-game map is right-aligned, leaving ~103px padding on left
    map_region_left_trim: int = 103

    # Coordinate system (matches Next.js app)
    coordinate_system_size: int = 1000
    slot_match_distance_threshold: int = 100
    # Offset to align OCR coordinates with frontend coordinate system
    # NOTE: Offset disabled - was causing issues for slots in different map areas
    # The misalignment may be non-uniform (scaling issue, not translation)
    coordinate_offset_x: int = 0
    coordinate_offset_y: int = 0

    # WebSocket settings
    ws_host: str = "0.0.0.0"
    ws_port: int = 8000

    # Default monitor for capture (can be changed per setup)
    default_monitor: int = 2

    # Paths
    templates_dir: str = "templates"

    class Config:
        env_prefix = "OCR_"


settings = Settings()


# Load boss data from JSON file
_BOSSES_JSON_PATH = Path(__file__).parent.parent / "data" / "bosses.json"


def _load_nightlord_mapping() -> dict:
    """Build NIGHTLORD_MAPPING from bosses.json template_names."""
    mapping = {}
    try:
        with open(_BOSSES_JSON_PATH, "r") as f:
            bosses_data = json.load(f)

        for nightlord_id, data in bosses_data.get("nightlords", {}).items():
            for template_name in data.get("template_names", []):
                mapping[template_name] = nightlord_id
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to hardcoded values if JSON fails to load
        mapping = {
            "Augur": "4_Maris",
            "GapingJaw": "2_Adel",
            "SentientPest": "3_Gnoster",
            "Tricephalos": "1_Gladius",
            "DarkdriftKnight": "6_Fulghor",
            "EquilibrousBeast": "5_Libra",
            "Balancers": "9_Balancers",
            "Dreglord": "10_Greg",
            "FissureInTheFog": "7_Caligo",
            "NightAspect": "8_Heolstor",
            "Libra": "5_Libra",
            "Caligo": "7_Caligo",
            "Heolstor": "8_Heolstor",
            "Greg": "10_Greg",
        }
    return mapping


# Nightlord mapping from in-game template names to app IDs
NIGHTLORD_MAPPING = _load_nightlord_mapping()

# Reverse mapping for display
NIGHTLORD_DISPLAY_NAMES = {v: k for k, v in NIGHTLORD_MAPPING.items()}

# Standard map slot coordinates (1000x1000 system)
BUILDING_SLOT_COORDINATES = [
    {"id": "1", "x": 400, "y": 180},
    {"id": "2", "x": 710, "y": 210},
    {"id": "3", "x": 535, "y": 225},
    {"id": "4", "x": 232, "y": 281},
    {"id": "5", "x": 628, "y": 293},
    {"id": "6", "x": 412, "y": 303},
    {"id": "7", "x": 776, "y": 361},
    {"id": "8", "x": 217, "y": 354},
    {"id": "9", "x": 693, "y": 370},
    {"id": "10", "x": 357, "y": 395},
    {"id": "11", "x": 580, "y": 430},
    {"id": "12", "x": 774, "y": 425},
    {"id": "13", "x": 282, "y": 447},
    {"id": "14", "x": 663, "y": 465},
    {"id": "15", "x": 318, "y": 550},
    {"id": "16", "x": 205, "y": 555},
    {"id": "17", "x": 804, "y": 576},
    {"id": "18", "x": 629, "y": 585},
    {"id": "19", "x": 550, "y": 630},
    {"id": "20", "x": 753, "y": 631},
    {"id": "21", "x": 276, "y": 650},
    {"id": "22", "x": 610, "y": 690},
    {"id": "23", "x": 452, "y": 695},
    {"id": "24", "x": 199, "y": 710},
    {"id": "25", "x": 745, "y": 740},
    {"id": "26", "x": 400, "y": 780},
    {"id": "27", "x": 566, "y": 795},
]

# Special coordinates (in 1000x1000 coordinate system)
# Nightlord icon appears in bottom-left of map overlay
NIGHTLORD_COORDINATE = {"id": "nightlord", "x": 163, "y": 800}
EVENT_COORDINATE = {"id": "event", "x": 102, "y": 740}
