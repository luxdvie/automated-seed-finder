"""FastAPI application for OCR seed detection service."""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import json
import numpy as np

from .config import settings, NIGHTLORD_MAPPING, NIGHTLORD_COORDINATE
from .capture import VideoCapture, FrameProcessor, screen_capture
from .detection import BossDetector, SpawnDetector, POIDetector, CoordinateMapper, ShiftingEarthDetector, FieldBossDetector
from .websocket import ConnectionManager
from .messaging.imessage import send_to_multiple, format_boss_message

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
field_boss_detector = FieldBossDetector()
coordinate_mapper = CoordinateMapper()
connection_manager = ConnectionManager()
overlay_manager = ConnectionManager()  # Separate manager for overlay WebSocket clients

# Track last detected bosses for "send current" functionality
last_detected_nightlord: Optional[str] = None
last_detected_field_boss: Optional[str] = None


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

# Calibration config path
CALIBRATION_FILE = Path(__file__).parent.parent / "calibration.json"

# SMS config path
SMS_CONFIG_FILE = Path(__file__).parent.parent / "sms_config.json"


def load_sms_config() -> dict:
    """Load SMS/iMessage config from file."""
    if SMS_CONFIG_FILE.exists():
        with open(SMS_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"enabled": False, "recipients": []}


def load_calibration_config() -> dict:
    """Load calibration config from file."""
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    return {"regions": {}}


def save_calibration_config(config: dict):
    """Save calibration config to file."""
    config["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(config, f, indent=2)


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


@app.get("/default-monitor")
async def get_default_monitor():
    """Get the configured default monitor index."""
    return {"default_monitor": settings.default_monitor}


@app.get("/capture-monitor")
@app.get("/capture-monitor/{monitor_index}")
async def capture_monitor(monitor_index: int = None, debug: bool = False, sms: bool = False):
    """Capture a monitor screenshot and analyze it.

    Args:
        monitor_index: Monitor to capture (1 = first monitor, 2 = second, etc.)
                      If not provided, uses the configured default_monitor.
        debug: If true, also save debug images
        sms: If true, send nightlord info via iMessage when detected

    This endpoint can be triggered by Stream Deck, curl, or any HTTP client:
        curl http://localhost:8000/capture-monitor
        curl http://localhost:8000/capture-monitor/2
        curl http://localhost:8000/capture-monitor?sms=true
    """
    from datetime import datetime
    from pathlib import Path

    if monitor_index is None:
        monitor_index = settings.default_monitor

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
            # Note: NIGHTLORD_COORDINATE was calibrated directly for the OCR extraction,
            # so it doesn't need the coordinate_offset applied
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

            # Save original full-screen capture
            original_path = debug_dir / f"capture_{timestamp}_original.jpg"
            cv2.imwrite(str(original_path), frame)

            # Save annotated debug capture
            debug_path = debug_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(debug_path), debug_image)
            result["debug_image"] = str(debug_path)
    except Exception as e:
        logger.warning(f"Failed to save debug image: {e}")

    # Broadcast to overlay WebSocket clients
    if overlay_manager.connection_count > 0:
        await overlay_manager.broadcast({
            "type": "detection",
            "nightlord": result.get("nightlord"),
            "nightlord_confidence": result.get("nightlord_confidence", 0),
            "shifting_earth": result.get("shifting_earth"),
            "shifting_earth_confidence": result.get("shifting_earth_confidence", 0),
        })

    # Track last detected nightlord for "send current" functionality
    global last_detected_nightlord
    if result.get("nightlord"):
        last_detected_nightlord = result["nightlord"]
        logger.info(f"Tracking last nightlord: {last_detected_nightlord}")

    # Send SMS if requested and nightlord detected
    if sms and result.get("nightlord"):
        sms_config = load_sms_config()
        if sms_config.get("enabled") and sms_config.get("recipients"):
            from .detection.field_boss_detector import get_all_boss_data
            boss_data = get_all_boss_data()
            nightlord_id = result["nightlord"]
            nightlord_info = boss_data.get("nightlords", {}).get(nightlord_id)

            if nightlord_info:
                boss_name = nightlord_info["names"][0] if nightlord_info.get("names") else nightlord_id
                message = format_boss_message(
                    boss_name=boss_name,
                    negations=nightlord_info.get("negations", {}),
                    status_resistances=nightlord_info.get("status_resistances", {}),
                    is_nightlord=True
                )
                sms_result = send_to_multiple(sms_config["recipients"], message)
                result["sms_sent"] = len(sms_result["success"])
                result["sms_failed"] = len(sms_result["failed"])
                logger.info(f"SMS sent for {nightlord_id}: {len(sms_result['success'])} success")

    return result


@app.get("/capture-field-boss")
@app.get("/capture-field-boss/{monitor_index}")
async def capture_field_boss(monitor_index: int = None, sms: bool = False):
    """Capture screen and detect field boss using OCR.

    Reads the boss name from the calibrated 'field_boss_name' region using OCR,
    then looks up the boss's weaknesses/resistances.

    Args:
        monitor_index: Monitor to capture. If not provided, uses configured default.
        sms: If true, send field boss info via iMessage when detected

    Trigger from Stream Deck:
        curl http://10.0.0.91:8000/capture-field-boss
        curl http://10.0.0.91:8000/capture-field-boss?sms=true
    """
    if monitor_index is None:
        monitor_index = settings.default_monitor

    # Check if field_boss_name region is calibrated
    calibration = load_calibration_config()
    cal_regions = calibration.get("regions", {})

    if "field_boss_name" not in cal_regions:
        return {
            "error": "field_boss_name region not calibrated",
            "hint": "Use /calibrate in Full Screen mode to draw a box around where field boss names appear"
        }

    # Capture full screen
    frame = screen_capture.capture_monitor(monitor_index)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    frame_h, frame_w = frame.shape[:2]

    # Extract the field boss name region
    region = cal_regions["field_boss_name"]
    x1 = int(region["x"] * frame_w)
    y1 = int(region["y"] * frame_h)
    x2 = int((region["x"] + region["width"]) * frame_w)
    y2 = int((region["y"] + region["height"]) * frame_h)

    boss_region = frame[y1:y2, x1:x2]

    if boss_region.size == 0:
        return {"error": "Empty boss region - check calibration"}

    # Create debug_captures directory
    debug_dir = Path(__file__).parent.parent / "debug_captures"
    debug_dir.mkdir(exist_ok=True)

    # Generate timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save original screenshot
    original_path = debug_dir / f"{timestamp}_original.jpg"
    cv2.imwrite(str(original_path), boss_region)

    # Run OCR detection
    result = field_boss_detector.detect(boss_region)

    # Create annotated version
    annotated = boss_region.copy()
    if result:
        label = result["boss_name"]
        color = (0, 255, 0)  # Green for success
    else:
        label = "NONE"
        color = (0, 0, 255)  # Red for failure

    # Add label to annotated image
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Save annotated version
    safe_label = label.replace(" ", "_").replace("'", "")
    annotated_path = debug_dir / f"{timestamp}_{safe_label}.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    logger.info(f"Field boss debug saved: {original_path.name}, {annotated_path.name}")

    if result:
        # Broadcast to overlay
        if overlay_manager.connection_count > 0:
            await overlay_manager.broadcast({
                "type": "field_boss",
                "boss_name": result["boss_name"],
                "negations": result["negations"],
                "confidence": result["confidence"]
            })

        # Track last detected field boss for "send current" functionality
        global last_detected_field_boss
        last_detected_field_boss = result["boss_name"]
        logger.info(f"Tracking last field boss: {last_detected_field_boss}")

        response = {
            "detected": True,
            "boss_name": result["boss_name"],
            "raw_text": result["raw_text"],
            "confidence": result["confidence"],
            "negations": result["negations"],
            "debug_original": str(original_path),
            "debug_annotated": str(annotated_path)
        }

        # Send SMS if requested
        if sms:
            sms_config = load_sms_config()
            if sms_config.get("enabled") and sms_config.get("recipients"):
                message = format_boss_message(
                    boss_name=result["boss_name"],
                    negations=result["negations"],
                    status_resistances=result.get("status_resistances", {}),
                    is_nightlord=False
                )
                sms_result = send_to_multiple(sms_config["recipients"], message)
                response["sms_sent"] = len(sms_result["success"])
                response["sms_failed"] = len(sms_result["failed"])
                logger.info(f"SMS sent for field boss {result['boss_name']}: {len(sms_result['success'])} success")

        return response
    else:
        # Broadcast error to overlay (shows red X for 3 seconds)
        if overlay_manager.connection_count > 0:
            await overlay_manager.broadcast({
                "type": "command",
                "command": "fieldBossError"
            })

        return {
            "detected": False,
            "hint": "No field boss name detected. Make sure boss name is visible on screen.",
            "debug_original": str(original_path),
            "debug_annotated": str(annotated_path)
        }


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
    Uses calibrated boss region if available.
    """
    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        return {"error": "Failed to extract map region"}

    region_height, region_width = map_region.shape[:2]

    # Use calibrated boss region if available
    calibration = load_calibration_config()
    cal_regions = calibration.get("regions", {})
    using_calibration = False

    search_region = map_region
    if "boss" in cal_regions:
        r = cal_regions["boss"]
        x1 = int(r["x"] * region_width)
        y1 = int(r["y"] * region_height)
        x2 = int((r["x"] + r["width"]) * region_width)
        y2 = int((r["y"] + r["height"]) * region_height)
        search_region = map_region[y1:y2, x1:x2]
        using_calibration = True
        cv2.imwrite("debug_boss_region.jpg", search_region)

    cv2.imwrite("debug_map_region.jpg", map_region)

    # Convert to grayscale
    if len(search_region.shape) == 3:
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = search_region

    all_scores = []

    if boss_detector and boss_detector.templates_loaded:
        for template_name in boss_detector.matcher.templates.keys():
            if "nightlords" in template_name:
                short_name = template_name.replace("nightlords/", "")

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
        "using_calibrated_region": using_calibration,
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

    # Load calibration config for region-based detection
    calibration = load_calibration_config()
    cal_regions = calibration.get("regions", {})

    # Run detections
    nightlord_result = None
    spawn_result = None
    buildings_result = []

    # Detect nightlord - use calibrated region if available
    if boss_detector is not None and boss_detector.templates_loaded:
        search_region = map_region
        if "boss" in cal_regions:
            # Extract the calibrated boss region
            r = cal_regions["boss"]
            x1 = int(r["x"] * region_width)
            y1 = int(r["y"] * region_height)
            x2 = int((r["x"] + r["width"]) * region_width)
            y2 = int((r["y"] + r["height"]) * region_height)
            search_region = map_region[y1:y2, x1:x2]
            logger.debug(f"Using calibrated boss region: ({x1},{y1}) to ({x2},{y2})")

        best_match = boss_detector.matcher.find_best_match(
            search_region, "nightlords", settings.template_match_threshold
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

    # Detect Shifting Earth event - searches entire map for templates
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


# =============================================================================
# CALIBRATION ENDPOINTS
# =============================================================================

@app.get("/calibrate")
async def calibrate_page():
    """Serve the calibration UI."""
    static_dir = Path(__file__).parent.parent / "static"
    html_file = static_dir / "calibrate.html"

    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Calibration page not found")

    with open(html_file, 'r') as f:
        return HTMLResponse(content=f.read())


@app.get("/overlay")
async def overlay_page():
    """Serve the OBS overlay page for boss weaknesses.

    URL params:
        monitor: Monitor index to poll (default: 2)
        poll: Poll interval in ms (default: 2000)
        boss: Show specific boss for testing (e.g., ?boss=10_Greg)

    Add this URL as a Browser Source in OBS:
        http://localhost:8000/overlay?monitor=2
    """
    static_dir = Path(__file__).parent.parent / "static"
    html_file = static_dir / "overlay.html"

    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Overlay page not found")

    with open(html_file, 'r') as f:
        return HTMLResponse(content=f.read())


@app.get("/api/bosses")
async def get_boss_data():
    """Get all boss data (nightlords and field bosses) from the central JSON file.

    Returns the complete boss database including negations, status resistances,
    names, and OCR spelling variations.
    """
    from .detection.field_boss_detector import get_all_boss_data
    return JSONResponse(content=get_all_boss_data())


@app.websocket("/ws/overlay")
async def overlay_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time overlay updates.

    Overlay clients connect here to receive detection broadcasts.
    """
    await overlay_manager.connect(websocket)
    logger.info(f"Overlay client connected. Total: {overlay_manager.connection_count}")

    try:
        while True:
            # Keep connection alive, wait for disconnect
            data = await websocket.receive_text()
            # Could handle ping/pong here if needed
    except Exception:
        pass
    finally:
        await overlay_manager.disconnect(websocket)
        logger.info(f"Overlay client disconnected. Total: {overlay_manager.connection_count}")


@app.get("/overlay-command")
async def overlay_command(command: str, boss: Optional[str] = None, fieldBoss: Optional[str] = None):
    """Send a command to all overlay clients.

    Commands:
        hide - Make overlay transparent/invisible
        show - Make overlay visible again
        reset - Reset overlay state (hides until next detection)
        showBoss - Show a specific nightlord (requires boss param)
        showFieldBoss - Show a specific field boss (requires fieldBoss param)
        hideFieldBoss - Hide the field boss panel
        fieldBossError - Show red X in top-left for 3 seconds (detection failed)

    Trigger from Stream Deck:
        http://10.0.0.91:8000/overlay-command?command=hide
        http://10.0.0.91:8000/overlay-command?command=show
        http://10.0.0.91:8000/overlay-command?command=reset
        http://10.0.0.91:8000/overlay-command?command=showBoss&boss=10_Greg
        http://10.0.0.91:8000/overlay-command?command=showFieldBoss&fieldBoss=Blackgaol%20Knight
        http://10.0.0.91:8000/overlay-command?command=hideFieldBoss
        http://10.0.0.91:8000/overlay-command?command=fieldBossError
    """
    valid_commands = ["hide", "show", "reset", "showBoss", "showFieldBoss", "hideFieldBoss", "fieldBossError"]
    if command not in valid_commands:
        return {"error": f"Invalid command. Valid: {valid_commands}"}

    if command == "showBoss" and not boss:
        return {"error": "showBoss requires boss parameter (e.g., boss=10_Greg)"}

    if command == "showFieldBoss" and not fieldBoss:
        return {"error": "showFieldBoss requires fieldBoss parameter (e.g., fieldBoss=Blackgaol%20Knight)"}

    if overlay_manager.connection_count == 0:
        return {"status": "no_clients", "message": "No overlay clients connected"}

    message = {"type": "command", "command": command}
    if boss:
        message["boss"] = boss
        # Track for "send current" functionality
        global last_detected_nightlord
        last_detected_nightlord = boss
        logger.info(f"Tracking last nightlord (via showBoss): {boss}")
    if fieldBoss:
        # Look up field boss data
        from .detection.field_boss_detector import FIELD_BOSS_DATA
        boss_data = FIELD_BOSS_DATA.get(fieldBoss)
        if not boss_data:
            return {"error": f"Unknown field boss: {fieldBoss}", "known_bosses": list(FIELD_BOSS_DATA.keys())}
        message["fieldBoss"] = fieldBoss
        message["negations"] = boss_data["negations"]
        # Track for "send current" functionality
        global last_detected_field_boss
        last_detected_field_boss = fieldBoss
        logger.info(f"Tracking last field boss (via showFieldBoss): {fieldBoss}")

    await overlay_manager.broadcast(message)

    logger.info(f"Overlay command sent: {command}" + (f" boss={boss}" if boss else "") + (f" fieldBoss={fieldBoss}" if fieldBoss else ""))
    return {"status": "ok", "command": command, "boss": boss, "fieldBoss": fieldBoss, "clients": overlay_manager.connection_count}


# =============================================================================
# iMESSAGE / SMS ENDPOINTS
# =============================================================================

@app.get("/send-nightlord-text")
@app.get("/send-nightlord-text/{boss_id}")
async def send_nightlord_text(boss_id: Optional[str] = None):
    """Send nightlord weaknesses/strengths via iMessage.

    Args:
        boss_id: Nightlord ID (e.g., "1_Gladius", "10_Greg").
                 If not provided, sends info for all nightlords (for testing).

    Trigger from Stream Deck:
        http://10.0.0.91:8000/send-nightlord-text/1_Gladius

    Configure recipients in sms_config.json
    """
    from .detection.field_boss_detector import get_all_boss_data

    # Load SMS config
    sms_config = load_sms_config()
    if not sms_config.get("enabled", False):
        return {"error": "SMS disabled", "hint": "Set enabled: true in sms_config.json"}

    recipients = sms_config.get("recipients", [])
    if not recipients:
        return {"error": "No recipients configured", "hint": "Add phone numbers to sms_config.json"}

    # Get boss data
    boss_data = get_all_boss_data()
    nightlords = boss_data.get("nightlords", {})

    if not boss_id:
        return {
            "error": "boss_id required",
            "available_bosses": list(nightlords.keys()),
            "example": "/send-nightlord-text/1_Gladius"
        }

    if boss_id not in nightlords:
        return {
            "error": f"Unknown boss: {boss_id}",
            "available_bosses": list(nightlords.keys())
        }

    # Get boss info
    boss_info = nightlords[boss_id]
    boss_name = boss_info["names"][0] if boss_info.get("names") else boss_id

    # Format message
    message = format_boss_message(
        boss_name=boss_name,
        negations=boss_info.get("negations", {}),
        status_resistances=boss_info.get("status_resistances", {}),
        is_nightlord=True
    )

    # Send to all recipients
    results = send_to_multiple(recipients, message)

    logger.info(f"Nightlord text sent: {boss_id} -> {len(results['success'])} success, {len(results['failed'])} failed")

    return {
        "status": "ok",
        "boss_id": boss_id,
        "boss_name": boss_name,
        "message_preview": message,
        "sent_to": results["success"],
        "failed": results["failed"]
    }


@app.get("/send-field-boss-text")
@app.get("/send-field-boss-text/{boss_name}")
async def send_field_boss_text(boss_name: Optional[str] = None):
    """Send field boss weaknesses/strengths via iMessage.

    Args:
        boss_name: Field boss name (e.g., "Bell Bearing Hunter").

    Trigger from Stream Deck:
        http://10.0.0.91:8000/send-field-boss-text/Bell%20Bearing%20Hunter

    Configure recipients in sms_config.json
    """
    from .detection.field_boss_detector import get_all_boss_data

    # Load SMS config
    sms_config = load_sms_config()
    if not sms_config.get("enabled", False):
        return {"error": "SMS disabled", "hint": "Set enabled: true in sms_config.json"}

    recipients = sms_config.get("recipients", [])
    if not recipients:
        return {"error": "No recipients configured", "hint": "Add phone numbers to sms_config.json"}

    # Get boss data
    boss_data = get_all_boss_data()
    field_bosses = boss_data.get("field_bosses", {})

    if not boss_name:
        return {
            "error": "boss_name required",
            "available_bosses": list(field_bosses.keys()),
            "example": "/send-field-boss-text/Bell%20Bearing%20Hunter"
        }

    if boss_name not in field_bosses:
        return {
            "error": f"Unknown field boss: {boss_name}",
            "available_bosses": list(field_bosses.keys())
        }

    # Get boss info
    boss_info = field_bosses[boss_name]

    # Format message
    message = format_boss_message(
        boss_name=boss_name,
        negations=boss_info.get("negations", {}),
        status_resistances=boss_info.get("status_resistances", {}),
        is_nightlord=False
    )

    # Send to all recipients
    results = send_to_multiple(recipients, message)

    logger.info(f"Field boss text sent: {boss_name} -> {len(results['success'])} success, {len(results['failed'])} failed")

    return {
        "status": "ok",
        "boss_name": boss_name,
        "message_preview": message,
        "sent_to": results["success"],
        "failed": results["failed"]
    }


@app.get("/send-current-nightlord-text")
async def send_current_nightlord_text():
    """Send iMessage for the currently displayed nightlord.

    Texts the weaknesses/strengths for whatever nightlord was last detected.
    Perfect for a single Stream Deck button that always sends the current boss info.

    Trigger from Stream Deck:
        http://10.0.0.91:8000/send-current-nightlord-text
    """
    global last_detected_nightlord

    if not last_detected_nightlord:
        return {
            "error": "No nightlord detected yet",
            "hint": "Detect a nightlord first with /capture-monitor"
        }

    # Delegate to the existing endpoint
    return await send_nightlord_text(last_detected_nightlord)


@app.get("/send-current-field-boss-text")
async def send_current_field_boss_text():
    """Send iMessage for the currently displayed field boss.

    Texts the weaknesses/strengths for whatever field boss was last detected.
    Perfect for a single Stream Deck button that always sends the current boss info.

    Trigger from Stream Deck:
        http://10.0.0.91:8000/send-current-field-boss-text
    """
    global last_detected_field_boss

    if not last_detected_field_boss:
        return {
            "error": "No field boss detected yet",
            "hint": "Detect a field boss first with /capture-field-boss"
        }

    # Delegate to the existing endpoint
    return await send_field_boss_text(last_detected_field_boss)


@app.get("/calibration/capture/{monitor_index}")
async def calibration_capture(monitor_index: int):
    """Capture the map region and return as image for calibration UI."""
    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    # Extract map region
    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        raise HTTPException(status_code=500, detail="Failed to extract map region")

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', map_region, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/calibration/capture-full/{monitor_index}")
async def calibration_capture_full(monitor_index: int):
    """Capture the full screen and return as image for calibration UI.

    Used for calibrating regions outside the map (e.g., field boss name text).
    """
    frame = screen_capture.capture_monitor(monitor_index)

    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    # Return full frame (not just map region)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/calibration/save")
async def calibration_save(data: dict):
    """Save calibration regions to config file.

    Merges new regions with existing ones (doesn't replace all).
    """
    try:
        # Load existing config and merge
        existing = load_calibration_config()
        existing_regions = existing.get("regions", {})

        # Merge new regions into existing
        new_regions = data.get("regions", {})
        existing_regions.update(new_regions)

        config = {
            "regions": existing_regions,
            "notes": "Coordinates are percentages (0-1) of the capture region"
        }
        save_calibration_config(config)
        logger.info(f"Calibration saved: {list(new_regions.keys())} (total: {list(existing_regions.keys())})")
        return {"status": "ok", "saved_regions": list(new_regions.keys()), "all_regions": list(existing_regions.keys())}
    except Exception as e:
        logger.error(f"Failed to save calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calibration/load")
async def calibration_load():
    """Load calibration regions from config file."""
    try:
        config = load_calibration_config()
        return config
    except Exception as e:
        logger.error(f"Failed to load calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calibration/save-template/{monitor_index}")
async def save_template_direct(monitor_index: int, data: dict):
    """Save a template directly from a drawn region.

    For nightlord templates: uses the calibrated boss region (not the drawn box).
    For other templates: uses the drawn box.

    Args:
        monitor_index: Monitor to capture from
        data: { category: "shifting_earth"|"nightlords", templateName: "Crater", region: {x, y, width, height} }
    """
    category = data.get("category")
    template_name = data.get("templateName")
    region = data.get("region")

    if not all([category, template_name, region]):
        return {"error": "Missing category, templateName, or region"}

    # Capture screen
    frame = screen_capture.capture_monitor(monitor_index)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        raise HTTPException(status_code=500, detail="Failed to extract map region")

    height, width = map_region.shape[:2]

    # For nightlord templates, use the calibrated boss region instead of the drawn box
    if category == "nightlords":
        calibration = load_calibration_config()
        cal_regions = calibration.get("regions", {})

        if "boss" not in cal_regions:
            return {"error": "No boss region calibrated. Please calibrate the boss region first."}

        r = cal_regions["boss"]
        x1 = int(r["x"] * width)
        y1 = int(r["y"] * height)
        x2 = int((r["x"] + r["width"]) * width)
        y2 = int((r["y"] + r["height"]) * height)
    else:
        # Use the drawn region for other templates (shifting earth, etc.)
        x1 = int(region["x"] * width)
        y1 = int(region["y"] * height)
        x2 = int((region["x"] + region["width"]) * width)
        y2 = int((region["y"] + region["height"]) * height)

    template = map_region[y1:y2, x1:x2]

    if template.size == 0:
        return {"error": "Empty template region"}

    # Save to appropriate directory
    save_dir = Path(__file__).parent.parent / "templates" / category
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{template_name}.png"

    cv2.imwrite(str(output_path), template)

    # Save debug image
    debug_region = map_region.copy()
    cv2.rectangle(debug_region, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(debug_region, f"{category}/{template_name}", (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("debug_template_extraction.jpg", debug_region)

    logger.info(f"Saved template: {output_path}")

    return {
        "message": f"Template saved",
        "saved_to": str(output_path),
        "template_size": {"width": x2 - x1, "height": y2 - y1}
    }


@app.post("/calibration/extract-template/{monitor_index}")
async def extract_template_from_calibration(monitor_index: int, region_name: str, template_name: str):
    """Extract a template from a calibrated region.

    Args:
        monitor_index: Monitor to capture from
        region_name: Name of the calibrated region (e.g., "shifting_earth")
        template_name: Name to save the template as (e.g., "Crater")
    """
    # Load calibration
    config = load_calibration_config()
    regions = config.get("regions", {})

    if region_name not in regions:
        return {"error": f"Region '{region_name}' not calibrated", "available": list(regions.keys())}

    # Capture screen
    frame = screen_capture.capture_monitor(monitor_index)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Failed to capture monitor {monitor_index}")

    map_region = frame_processor.extract_map_region(frame)
    if map_region is None:
        raise HTTPException(status_code=500, detail="Failed to extract map region")

    height, width = map_region.shape[:2]

    # Extract the calibrated region
    r = regions[region_name]
    x1 = int(r["x"] * width)
    y1 = int(r["y"] * height)
    x2 = int((r["x"] + r["width"]) * width)
    y2 = int((r["y"] + r["height"]) * height)

    template = map_region[y1:y2, x1:x2]

    if template.size == 0:
        return {"error": "Empty template region"}

    # Determine save path based on region type
    region_type = r.get("type", region_name)
    if region_type == "shifting_earth":
        save_dir = Path(__file__).parent.parent / "templates" / "shifting_earth"
    elif region_type == "boss":
        save_dir = Path(__file__).parent.parent / "templates" / "nightlords"
    else:
        save_dir = Path(__file__).parent.parent / "templates" / region_type

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{template_name}.png"

    cv2.imwrite(str(output_path), template)

    # Save debug image showing extraction
    debug_region = map_region.copy()
    cv2.rectangle(debug_region, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(debug_region, f"{region_name}: {template_name}", (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("debug_template_extraction.jpg", debug_region)

    return {
        "message": f"Extracted template '{template_name}' from region '{region_name}'",
        "saved_to": str(output_path),
        "template_size": {"width": x2 - x1, "height": y2 - y1},
        "hint": "Restart the service to reload templates"
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
