"""FastAPI application for OCR seed detection service."""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from .config import settings, NIGHTLORD_MAPPING, NIGHTLORD_COORDINATE
from .capture import VideoCapture, FrameProcessor, screen_capture
from .detection import BossDetector, SpawnDetector, POIDetector, CoordinateMapper, ShiftingEarthDetector
from .websocket import ConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
video_capture: Optional[VideoCapture] = None
frame_processor = FrameProcessor()
boss_detector: Optional[BossDetector] = None
spawn_detector = SpawnDetector()
poi_detector: Optional[POIDetector] = None
shifting_earth_detector: Optional[ShiftingEarthDetector] = None
coordinate_mapper = CoordinateMapper()
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global boss_detector, poi_detector, shifting_earth_detector

    # Initialize detectors and load templates
    templates_dir = Path(__file__).parent.parent / "templates"

    boss_detector = BossDetector(str(templates_dir))
    poi_detector = POIDetector(str(templates_dir))
    shifting_earth_detector = ShiftingEarthDetector(str(templates_dir))

    # Try to load templates (will log warnings if not found)
    boss_detector.load_templates()
    poi_detector.load_templates()
    shifting_earth_detector.load_templates()

    logger.info("OCR service started")

    yield

    # Cleanup
    if video_capture is not None:
        video_capture.stop()

    logger.info("OCR service stopped")


app = FastAPI(
    title="OCR Seed Detection Service",
    description="Video capture and OCR detection for game seed finding",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://10.0.0.91:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "ocr-seed-detection"}


@app.get("/devices")
async def list_devices():
    """List available capture devices."""
    devices = VideoCapture.list_devices()
    return {"devices": devices}


@app.get("/debug-capture/{device_index}")
async def debug_capture(device_index: int):
    """Capture a frame and save it for debugging.

    Saves:
    - debug_frame.jpg: The raw captured frame
    - debug_map_region.jpg: The extracted map region
    """
    cap = VideoCapture(device_index)
    if not cap.start():
        raise HTTPException(status_code=500, detail=f"Failed to start capture on device {device_index}")

    frame = cap.capture_screenshot()
    cap.stop()

    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame")

    # Save raw frame
    cv2.imwrite("debug_frame.jpg", frame)

    # Extract and save map region
    map_region = frame_processor.extract_map_region(frame)
    if map_region is not None:
        cv2.imwrite("debug_map_region.jpg", map_region)

    return {
        "message": "Debug images saved",
        "frame_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "map_region_size": {"width": map_region.shape[1], "height": map_region.shape[0]} if map_region is not None else None,
        "files": ["debug_frame.jpg", "debug_map_region.jpg"]
    }


@app.get("/monitors")
async def list_monitors():
    """List all available monitors for screen capture."""
    monitors = screen_capture.list_monitors()
    return {"monitors": monitors}


@app.get("/capture-monitor/{monitor_index}")
async def capture_monitor(monitor_index: int, debug: bool = False):
    """Capture a monitor screenshot and analyze it.

    Args:
        monitor_index: Monitor to capture (1 = first monitor, 2 = second, etc.)
        debug: If true, also save debug images

    This endpoint can be triggered by Stream Deck, curl, or any HTTP client:
        curl http://localhost:8000/capture-monitor/2
    """
    from datetime import datetime
    from pathlib import Path

    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to capture monitor {monitor_index}"
        )

    if debug:
        cv2.imwrite("debug_frame.jpg", frame)
        map_region = frame_processor.extract_map_region(frame)
        if map_region is not None:
            cv2.imwrite("debug_map_region.jpg", map_region)

    # Process and return detection results
    result = await process_frame(frame)

    # Save annotated debug image with timestamp
    try:
        map_region = frame_processor.extract_map_region(frame)
        if map_region is not None:
            debug_image = map_region.copy()
            height, width = map_region.shape[:2]

            # Draw GREEN box on nightlord position
            nl_x = NIGHTLORD_COORDINATE["x"]
            nl_y = NIGHTLORD_COORDINATE["y"]
            scale_x = width / 1000
            scale_y = height / 1000
            nl_px = int(nl_x * scale_x)
            nl_py = int(nl_y * scale_y)
            box_size = int(min(width, height) * 0.05)
            cv2.rectangle(debug_image,
                          (nl_px - box_size, nl_py - box_size),
                          (nl_px + box_size, nl_py + box_size),
                          (0, 255, 0), 3)  # Green

            # Add nightlord label
            nl_label = result.get("nightlord_template") or result.get("nightlord") or "none"
            nl_conf = result.get("nightlord_confidence", 0)
            cv2.putText(debug_image, f"{nl_label} ({nl_conf:.0%})",
                        (nl_px - box_size, nl_py - box_size - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw YELLOW circle on spawn position
            spawn_debug = result.get("spawn_debug")
            if spawn_debug and spawn_debug.get("pixel"):
                sp_x = spawn_debug["pixel"]["x"]
                sp_y = spawn_debug["pixel"]["y"]
                cv2.circle(debug_image, (sp_x, sp_y), 25, (0, 255, 255), 3)  # Yellow

                # Add spawn label
                spawn_slot = result.get("spawn_slot") or "none"
                spawn_conf = result.get("spawn_confidence", 0)
                cv2.putText(debug_image, f"Slot {spawn_slot} ({spawn_conf:.0%})",
                            (sp_x - 50, sp_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Add CYAN text for Shifting Earth in top-right corner
            shifting_earth = result.get("shifting_earth")
            se_conf = result.get("shifting_earth_confidence", 0)
            if shifting_earth:
                se_label = f"Shifting Earth: {shifting_earth} ({se_conf:.0%})"
            else:
                se_label = "No Shifting Earth"
            # Position in top-right, with some padding
            text_size = cv2.getTextSize(se_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = width - text_size[0] - 20
            text_y = 40
            cv2.putText(debug_image, se_label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Cyan

            # Save with timestamp
            debug_dir = Path(__file__).parent.parent / "debug_captures"
            debug_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = debug_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(debug_path), debug_image)
            result["debug_image"] = str(debug_path)
    except Exception as e:
        logger.warning(f"Failed to save debug image: {e}")

    return result


@app.get("/debug-detection/{monitor_index}")
async def debug_detection(monitor_index: int):
    """Detailed debug info for detection issues.

    Saves debug images and returns verbose matching info.
    """
    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    # Save full frame
    cv2.imwrite("debug_frame.jpg", frame)
    frame_h, frame_w = frame.shape[:2]

    # Extract map region
    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        return {"error": "Failed to extract map region"}

    cv2.imwrite("debug_map_region.jpg", map_region)
    region_h, region_w = map_region.shape[:2]

    # Check loaded templates
    loaded_templates = list(boss_detector.matcher.templates.keys()) if boss_detector else []
    template_sizes = {}
    if boss_detector:
        for name, tmpl in boss_detector.matcher.templates.items():
            template_sizes[name] = {"height": tmpl.shape[0], "width": tmpl.shape[1]}

    # Convert map region to grayscale for matching
    if len(map_region.shape) == 3:
        gray_region = cv2.cvtColor(map_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_region = map_region

    # Try matching each template directly to see raw scores
    template_scores = []
    match_errors = []
    raw_scores = []

    if boss_detector and boss_detector.templates_loaded:
        for template_name in loaded_templates:
            if "nightlords" in template_name:
                try:
                    template = boss_detector.matcher.templates[template_name]

                    # Try direct match at scale 1.0
                    result = cv2.matchTemplate(gray_region, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    raw_scores.append({
                        "template": template_name,
                        "max_val": float(max_val),
                        "min_val": float(min_val),
                        "max_loc": {"x": int(max_loc[0]), "y": int(max_loc[1])}
                    })

                    # Also try multi-scale
                    multi_result = boss_detector.matcher.match_template_multi_scale(
                        gray_region, template_name, threshold=0.0
                    )
                    if multi_result:
                        x, y, confidence, scale = multi_result
                        template_scores.append({
                            "template": template_name,
                            "confidence": confidence,
                            "position": {"x": x, "y": y},
                            "scale": scale
                        })
                except Exception as e:
                    match_errors.append({"template": template_name, "error": str(e)})

    # Sort by confidence
    template_scores.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "frame_size": {"width": frame_w, "height": frame_h},
        "map_region_size": {"width": region_w, "height": region_h},
        "map_region_config": {
            "x_start": settings.map_region_x_start,
            "x_end": settings.map_region_x_end,
            "y_start": settings.map_region_y_start,
            "y_end": settings.map_region_y_end,
        },
        "templates_loaded": loaded_templates,
        "template_sizes": template_sizes,
        "template_match_threshold": settings.template_match_threshold,
        "template_scores": template_scores[:10],  # Top 10 matches
        "raw_scores": sorted(raw_scores, key=lambda x: x["max_val"], reverse=True),
        "match_errors": match_errors,
        "debug_files": ["debug_frame.jpg", "debug_map_region.jpg"]
    }


@app.get("/extract-nightlord/{monitor_index}")
async def extract_nightlord_template(monitor_index: int, name: str):
    """Extract nightlord template from current screen.

    The nightlord should be visible on the map. This extracts from the expected
    nightlord position (bottom-left of map area).

    Args:
        monitor_index: Monitor to capture
        name: Nightlord name (e.g., "Tricephalos")
    """
    if name not in NIGHTLORD_MAPPING:
        return {"error": f"Unknown nightlord: {name}", "valid_names": list(NIGHTLORD_MAPPING.keys())}

    frame = screen_capture.capture_monitor(monitor_index)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        return {"error": "Failed to extract map region"}

    from .config import NIGHTLORD_COORDINATE

    height, width = map_region.shape[:2]

    # Nightlord position in 1000x1000 system (from config)
    nl_x = NIGHTLORD_COORDINATE["x"]
    nl_y = NIGHTLORD_COORDINATE["y"]
    scale_x = width / 1000
    scale_y = height / 1000

    center_x = int(nl_x * scale_x)
    center_y = int(nl_y * scale_y)

    # Extract a region around the nightlord
    icon_size = int(min(width, height) * 0.10)  # 10% of map size (~130px)
    half = icon_size // 2

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(width, center_x + half)
    y2 = min(height, center_y + half)

    template = map_region[y1:y2, x1:x2]

    if template.size == 0:
        return {"error": "Empty template region"}

    # Save template
    from pathlib import Path
    templates_dir = Path(__file__).parent.parent / "templates" / "nightlords"
    templates_dir.mkdir(parents=True, exist_ok=True)

    output_path = templates_dir / f"{name}.png"
    cv2.imwrite(str(output_path), template)

    # Also save debug images
    cv2.imwrite("debug_map_region.jpg", map_region)

    # Draw rectangle showing where we extracted from
    debug_region = map_region.copy()
    cv2.rectangle(debug_region, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("debug_extraction_region.jpg", debug_region)

    return {
        "message": f"Extracted template for {name}",
        "saved_to": str(output_path),
        "template_size": {"width": x2 - x1, "height": y2 - y1},
        "extraction_region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "map_size": {"width": width, "height": height},
        "hint": "Restart the service to reload templates, then test with /all-scores/{monitor}"
    }


@app.get("/extract-shifting-earth/{monitor_index}")
async def extract_shifting_earth_template(monitor_index: int, name: str):
    """Extract Shifting Earth template from current screen.

    The shifting earth event should be visible on the map.

    Args:
        monitor_index: Monitor to capture
        name: Event name (MountainTop, Crater, Noklateo, RottedWoods, GreatHollow)
    """
    from .detection.shifting_earth_detector import SHIFTING_EARTH_REGIONS

    valid_names = list(SHIFTING_EARTH_REGIONS.keys())
    if name not in valid_names:
        return {"error": f"Unknown shifting earth: {name}", "valid_names": valid_names}

    frame = screen_capture.capture_monitor(monitor_index)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        return {"error": "Failed to extract map region"}

    height, width = map_region.shape[:2]

    # Get the region for this event type
    region_info = SHIFTING_EARTH_REGIONS[name]
    x1 = int(region_info["x_start"] * width)
    x2 = int(region_info["x_end"] * width)
    y1 = int(region_info["y_start"] * height)
    y2 = int(region_info["y_end"] * height)

    # Extract the region as template
    template = map_region[y1:y2, x1:x2]

    if template.size == 0:
        return {"error": "Empty template region"}

    # Save template
    from pathlib import Path
    templates_dir = Path(__file__).parent.parent / "templates" / "shifting_earth"
    templates_dir.mkdir(parents=True, exist_ok=True)

    output_path = templates_dir / f"{name}.png"
    cv2.imwrite(str(output_path), template)

    # Save debug images
    cv2.imwrite("debug_map_region.jpg", map_region)

    # Draw rectangle showing extraction region
    debug_region = map_region.copy()
    cv2.rectangle(debug_region, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(debug_region, name, (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("debug_extraction_region.jpg", debug_region)

    return {
        "message": f"Extracted template for {name}",
        "saved_to": str(output_path),
        "template_size": {"width": x2 - x1, "height": y2 - y1},
        "extraction_region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "map_size": {"width": width, "height": height},
        "hint": "Restart the service to reload templates, then test with /debug-shifting-earth/{monitor}"
    }


@app.get("/debug-shifting-earth/{monitor_index}")
async def debug_shifting_earth(monitor_index: int):
    """Debug Shifting Earth detection - shows all scores."""
    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        return {"error": "Failed to extract map region"}

    cv2.imwrite("debug_map_region.jpg", map_region)

    # Get all detection scores
    all_scores = []
    if shifting_earth_detector and shifting_earth_detector.templates_loaded:
        all_scores = shifting_earth_detector.detect_all(map_region)

    # Get the best detection
    best = shifting_earth_detector.detect(map_region) if shifting_earth_detector else None

    return {
        "detected": best,
        "all_scores": all_scores,
        "templates_loaded": shifting_earth_detector.templates_loaded if shifting_earth_detector else False,
        "hint": "If no templates loaded, use /extract-shifting-earth/{monitor}?name=... to create them"
    }


@app.get("/all-scores/{monitor_index}")
async def all_template_scores(monitor_index: int):
    """Get all nightlord template match scores for current screen.

    Returns sorted list of all template matches to help debug which is being detected.
    """
    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        return {"error": "Failed to extract map region"}

    cv2.imwrite("debug_map_region.jpg", map_region)

    # Convert to grayscale
    if len(map_region.shape) == 3:
        gray = cv2.cvtColor(map_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = map_region

    all_scores = []

    if boss_detector and boss_detector.templates_loaded:
        for template_name in boss_detector.matcher.templates.keys():
            if "nightlords" in template_name:
                short_name = template_name.replace("nightlords/", "")
                template = boss_detector.matcher.templates[template_name]

                # Try multi-scale matching with threshold=0 to get all scores
                result = boss_detector.matcher.match_template_multi_scale(
                    gray, template_name, threshold=0.0
                )
                if result:
                    x, y, confidence, scale = result
                    all_scores.append({
                        "template": short_name,
                        "confidence": round(confidence, 4),
                        "scale": scale,
                        "position": {"x": x, "y": y},
                        "maps_to": NIGHTLORD_MAPPING.get(short_name, "unknown")
                    })

    # Sort by confidence descending
    all_scores.sort(key=lambda x: x["confidence"], reverse=True)

    # Determine what would be selected
    threshold = settings.template_match_threshold
    selected = None
    for score in all_scores:
        if score["confidence"] >= threshold:
            selected = score
            break

    return {
        "threshold": threshold,
        "selected": selected,
        "all_scores": all_scores,
        "hint": "Compare the top scores - if wrong template is being selected, templates may need re-extraction"
    }


@app.get("/debug-spawn-visual/{monitor_index}")
async def debug_spawn_visual(monitor_index: int):
    """Visual debug for spawn detection - draws circle on detected spawn."""
    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        return {"error": "Failed to extract map region"}

    # Run spawn detection
    detection = spawn_detector.detect(map_region)
    multiple = spawn_detector.detect_multiple(map_region, max_count=5)

    # Draw on the map region
    debug_image = map_region.copy()

    # Draw all candidates as small yellow circles
    for candidate in multiple:
        cx = candidate["position"]["x"]
        cy = candidate["position"]["y"]
        cv2.circle(debug_image, (cx, cy), 15, (0, 255, 255), 2)  # Yellow

    # Draw the best detection as a larger green circle
    if detection:
        cx = detection["position"]["x"]
        cy = detection["position"]["y"]
        cv2.circle(debug_image, (cx, cy), 25, (0, 255, 0), 3)  # Green
        cv2.putText(debug_image, "SPAWN", (cx - 30, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite("debug_spawn_visual.jpg", debug_image)

    # Also save the map region without annotations
    cv2.imwrite("debug_map_region.jpg", map_region)

    height, width = map_region.shape[:2]
    spawn_info = None
    if detection:
        px, py = detection["position"]["x"], detection["position"]["y"]
        sx = int(px * 1000 / width)
        sy = int(py * 1000 / height)
        spawn_info = {
            "pixel": {"x": px, "y": py},
            "system": {"x": sx, "y": sy},
            "confidence": detection["confidence"]
        }

    return {
        "detection": spawn_info,
        "candidates_count": len(multiple),
        "debug_files": [
            "debug_spawn_visual.jpg - green circle = best detection, yellow = other candidates",
            "debug_map_region.jpg - original map"
        ]
    }


@app.get("/debug-spawn/{monitor_index}")
async def debug_spawn(monitor_index: int):
    """Debug spawn marker detection.

    Saves color mask images to help tune HSV values.
    """
    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        return {"error": "Failed to extract map region"}

    cv2.imwrite("debug_map_region.jpg", map_region)

    # Convert to HSV and create masks
    hsv = cv2.cvtColor(map_region, cv2.COLOR_BGR2HSV)

    # Save HSV channels separately for analysis
    h, s, v = cv2.split(hsv)
    cv2.imwrite("debug_hue.jpg", h)
    cv2.imwrite("debug_saturation.jpg", s)
    cv2.imwrite("debug_value.jpg", v)

    # Try VERY broad ranges to see what's being picked up
    # Broad red (any red hue, any saturation > 50, any value > 50)
    broad_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    broad_red2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    broad_red = cv2.bitwise_or(broad_red1, broad_red2)

    # Broad green
    broad_green = cv2.inRange(hsv, np.array([30, 50, 50]), np.array([90, 255, 255]))

    # Broad blue
    broad_blue = cv2.inRange(hsv, np.array([85, 50, 50]), np.array([135, 255, 255]))

    # Very saturated colors only (S > 100)
    saturated_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([180, 255, 255]))

    cv2.imwrite("debug_broad_red.jpg", broad_red)
    cv2.imwrite("debug_broad_green.jpg", broad_green)
    cv2.imwrite("debug_broad_blue.jpg", broad_blue)
    cv2.imwrite("debug_saturated.jpg", saturated_mask)

    # Try detection with current settings
    detection = spawn_detector.detect(map_region)
    multiple = spawn_detector.detect_multiple(map_region, max_count=10)

    return {
        "detection": detection,
        "multiple_candidates": multiple,
        "debug_files": [
            "debug_map_region.jpg - original map",
            "debug_hue.jpg - H channel (color)",
            "debug_saturation.jpg - S channel (color intensity)",
            "debug_value.jpg - V channel (brightness)",
            "debug_broad_red.jpg - broad red detection",
            "debug_broad_green.jpg - broad green detection",
            "debug_broad_blue.jpg - broad blue detection",
            "debug_saturated.jpg - any saturated color"
        ],
        "hint": "Check debug_saturated.jpg first - if spawn marker appears there, we can tune the specific color"
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image for seed detection.

    Args:
        file: Uploaded image file.

    Returns:
        Detection results.
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process the image
        result = await process_frame(image)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_frame(frame: np.ndarray) -> dict:
    """Process a frame and run all detections.

    Args:
        frame: Input frame/image.

    Returns:
        Detection results dictionary.
    """
    # Extract map region
    map_region = frame_processor.extract_map_region(frame)

    if map_region is None:
        return {
            "timestamp": time.time(),
            "nightlord": None,
            "nightlord_confidence": 0,
            "spawn_slot": None,
            "spawn_confidence": 0,
            "buildings": [],
            "error": "Failed to extract map region"
        }

    region_height, region_width = map_region.shape[:2]

    # Run detections
    nightlord_result = None
    spawn_result = None
    buildings_result = []

    # Detect nightlord - search full map region for best match
    if boss_detector is not None and boss_detector.templates_loaded:
        best_match = boss_detector.matcher.find_best_match(
            map_region, "nightlords", settings.template_match_threshold
        )
        if best_match:
            template_name = best_match["template"]
            # Map template name to app nightlord ID
            nightlord_id = NIGHTLORD_MAPPING.get(template_name)
            if nightlord_id:
                nightlord_result = {
                    "nightlord": nightlord_id,
                    "confidence": best_match["confidence"],
                    "template_name": template_name
                }

    # Detect spawn
    spawn_detection = spawn_detector.detect(map_region)
    spawn_debug = None
    if spawn_detection:
        pixel_x = spawn_detection["position"]["x"]
        pixel_y = spawn_detection["position"]["y"]
        system_x, system_y = coordinate_mapper.pixel_to_system(
            pixel_x, pixel_y, region_width, region_height
        )
        spawn_debug = {
            "pixel": {"x": pixel_x, "y": pixel_y},
            "system": {"x": system_x, "y": system_y},
            "region_size": {"width": region_width, "height": region_height}
        }
        logger.info(f"Spawn detected: pixel=({pixel_x}, {pixel_y}) -> system=({system_x}, {system_y})")

        spawn_mapped = coordinate_mapper.map_detection_to_slot(
            spawn_detection, region_width, region_height
        )
        if spawn_mapped:
            spawn_result = spawn_mapped
            logger.info(f"Spawn mapped to slot: {spawn_mapped['slot_id']}")
        else:
            # Find nearest slots for debugging
            nearby = coordinate_mapper.find_all_slots_within_distance(system_x, system_y, 200)
            spawn_debug["nearby_slots"] = nearby[:5]
            logger.warning(f"Spawn not mapped - nearest slots: {nearby[:3]}")

    # Detect buildings (POIs)
    if poi_detector is not None:
        poi_detections = poi_detector.detect(map_region)
        buildings_result = coordinate_mapper.map_detections_to_slots(
            poi_detections, region_width, region_height
        )

    # Detect Shifting Earth event
    shifting_earth_result = None
    if shifting_earth_detector is not None and shifting_earth_detector.templates_loaded:
        se_detection = shifting_earth_detector.detect(map_region)
        if se_detection:
            shifting_earth_result = {
                "event": se_detection["event"],
                "event_name": se_detection["event_name"],
                "confidence": se_detection["confidence"]
            }
            logger.info(f"Shifting Earth detected: {se_detection['event_name']} ({se_detection['confidence']:.0%})")

    return {
        "timestamp": time.time(),
        "nightlord": nightlord_result["nightlord"] if nightlord_result else None,
        "nightlord_confidence": nightlord_result["confidence"] if nightlord_result else 0,
        "nightlord_template": nightlord_result["template_name"] if nightlord_result else None,
        "spawn_slot": spawn_result["slot_id"] if spawn_result else None,
        "spawn_confidence": spawn_result["confidence"] if spawn_result else 0,
        "spawn_debug": spawn_debug,
        "shifting_earth": shifting_earth_result["event"] if shifting_earth_result else None,
        "shifting_earth_confidence": shifting_earth_result["confidence"] if shifting_earth_result else 0,
        "buildings": [
            {
                "slot_id": b["slot_id"],
                "building_type": b.get("building_type"),
                "confidence": b.get("confidence", 0)
            }
            for b in buildings_result
        ]
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time detection."""
    global video_capture

    await connection_manager.connect(websocket)
    await connection_manager.send_status(websocket, "connected", "Connected to OCR service")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "start_capture":
                # Start video capture
                device_index = data.get("data", {}).get("device_index", 0)

                if video_capture is not None:
                    video_capture.stop()

                video_capture = VideoCapture(device_index)
                success = video_capture.start()

                if success:
                    await connection_manager.send_status(
                        websocket, "capturing",
                        f"Started capture on device {device_index}"
                    )
                else:
                    await connection_manager.send_error(
                        websocket,
                        f"Failed to start capture on device {device_index}",
                        "CAPTURE_FAILED"
                    )

            elif message_type == "stop_capture":
                # Stop video capture
                if video_capture is not None:
                    video_capture.stop()
                    video_capture = None
                    await connection_manager.send_status(
                        websocket, "stopped", "Capture stopped"
                    )

            elif message_type == "capture_screenshot":
                # Capture and analyze a single frame
                if video_capture is None or not video_capture.is_capturing:
                    await connection_manager.send_error(
                        websocket,
                        "Capture not running. Start capture first.",
                        "NOT_CAPTURING"
                    )
                    continue

                frame = video_capture.capture_screenshot()
                if frame is None:
                    await connection_manager.send_error(
                        websocket, "Failed to capture frame", "FRAME_FAILED"
                    )
                    continue

                # Process frame
                result = await process_frame(frame)

                # Send detection result
                await connection_manager.send_detection_result(
                    websocket,
                    result.get("nightlord"),
                    result.get("nightlord_confidence", 0),
                    result.get("spawn_slot"),
                    result.get("spawn_confidence", 0),
                    result.get("buildings", []),
                    result.get("timestamp")
                )

            elif message_type == "ping":
                # Keep-alive ping
                await connection_manager.send_personal_message(
                    {"type": "pong", "data": {"timestamp": time.time()}},
                    websocket
                )

            else:
                await connection_manager.send_error(
                    websocket,
                    f"Unknown message type: {message_type}",
                    "UNKNOWN_TYPE"
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.ws_host,
        port=settings.ws_port,
        reload=True
    )
