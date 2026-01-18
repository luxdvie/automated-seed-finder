# OCR Seed Detection Service

Python backend service for video capture and OCR-based seed detection.

## Setup

1. Create a virtual environment:
```bash
cd ocr-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Extract templates from example screenshots:
```bash
# Auto-extract nightlord templates from examples
python scripts/extract_templates.py \
  --input-dir ../ocr/examples \
  --output-dir templates/nightlords \
  --auto-detect nightlord

# Or manually extract specific regions
python scripts/extract_templates.py \
  --input ../ocr/examples/Augur-NoShiftingEarth.jpeg \
  --output templates/nightlords/Augur.png \
  --interactive
```

## Running the Service

Start the FastAPI server:
```bash
python -m uvicorn app.main:app --reload --port 8000
```

Or run directly:
```bash
python -m app.main
```

The service will be available at:
- REST API: http://localhost:8000
- WebSocket: ws://localhost:8000/ws
- API docs: http://localhost:8000/docs

## API Endpoints

### REST

- `GET /` - Health check
- `GET /devices` - List available capture devices
- `POST /analyze-image` - Upload and analyze a single image

### WebSocket Messages

**Client -> Server:**

```json
{ "type": "start_capture", "data": { "device_index": 0 } }
{ "type": "stop_capture" }
{ "type": "capture_screenshot" }
{ "type": "ping" }
```

**Server -> Client:**

```json
{
  "type": "detection_result",
  "data": {
    "timestamp": 1234567890.123,
    "nightlord": "1_Gladius",
    "nightlord_confidence": 0.95,
    "spawn_slot": "15",
    "spawn_confidence": 0.82,
    "buildings": [
      { "slot_id": "5", "building_type": "forge", "confidence": 0.88 }
    ]
  }
}
```

```json
{
  "type": "status",
  "data": { "status": "connected", "message": "Connected to OCR service" }
}
```

```json
{
  "type": "error",
  "data": { "error": "Error message", "code": "ERROR_CODE" }
}
```

## Directory Structure

```
ocr-service/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings
│   ├── capture/
│   │   ├── video_capture.py # HDMI capture
│   │   └── frame_processor.py
│   ├── detection/
│   │   ├── template_matcher.py
│   │   ├── boss_detector.py
│   │   ├── spawn_detector.py
│   │   ├── poi_detector.py
│   │   └── coordinate_mapper.py
│   └── websocket/
│       └── manager.py
├── templates/               # Template images
│   ├── nightlords/
│   ├── buildings/
│   └── markers/
├── scripts/
│   └── extract_templates.py
├── requirements.txt
└── README.md
```

## Configuration

Environment variables (prefix with `OCR_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_CAPTURE_DEVICE_INDEX` | 0 | Video capture device index |
| `OCR_CAPTURE_WIDTH` | 1920 | Capture width |
| `OCR_CAPTURE_HEIGHT` | 1080 | Capture height |
| `OCR_TEMPLATE_MATCH_THRESHOLD` | 0.75 | Min confidence for matches |
| `OCR_WS_PORT` | 8000 | WebSocket server port |
