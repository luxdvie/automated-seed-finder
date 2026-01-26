# Detection module
from .template_matcher import TemplateMatcher
from .boss_detector import BossDetector
from .spawn_detector import SpawnDetector
from .poi_detector import POIDetector
from .coordinate_mapper import CoordinateMapper
from .shifting_earth_detector import ShiftingEarthDetector
from .field_boss_detector import FieldBossDetector

__all__ = [
    "TemplateMatcher",
    "BossDetector",
    "SpawnDetector",
    "POIDetector",
    "CoordinateMapper",
    "ShiftingEarthDetector",
    "FieldBossDetector",
]
