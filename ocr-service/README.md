# OCR Seed Detection Service

Python backend service for video capture and OCR-based seed detection.

## Setup Guide

- Expected Usage
  - `npm run build && npm run dev` in one terminal to run the base next app
    - Open http://127.0.0.1:3000/ and refresh a few times and make sure it loads
  - `npm run ocr` in another terminal to run the ocr app
    - See detailed first-time install instructions below
  - Uses:
    - Elgato Capture 4ks
    - PS5 -> Input -> Output to Monitor
  - Use OBS
    - 4ks is video input
      - 1920x1080 60fps
    - 4ks is audio input
      - In OBS -> Go to "Advanced Audio Properties" -> Audio Input Capture -> "Monitor and Output" to hear the ps5
    - Browser is the top-most option too
      - URL: http://127.0.0.1:8000/overlay
      - 1920x1080
  - Use a stream deck:
    - When you land, open the map and pause for a sec:
    - Press elgato stream deck buttons for URLs:
      - When at church: http://127.0.0.1:3000/capture?church=true
      - When not at church: http://127.0.0.1:3000/capture
      - Then it will navigate you as deep into the seed finder as it can in a new browser window
        - You will probably have to choose one or two more PoI's to figure out seed
    - When a field boss fight starts:
      - Press Stream Deck button with:
        - http://127.0.0.1:8000/capture-field-boss
    - Other commands:
      - http://127.0.0.1:8000/overlay-command?command=hide (hides everything, but keeps it in memory)
      - http://127.0.0.1:8000/overlay-command?command=show (shows whatever was hidden)
      - http://127.0.0.1:8000/overlay-command?command=reset (full clear the overlay)
	  - http://127.0.0.1:8000/overlay-command?command=hideFieldBoss (hides just the field boss)
	  - http://127.0.0.1:8000/overlay-command?command=showFieldBoss&fieldBoss=Blackgaol%20Knight (test field boss overlay)
	  - http://127.0.0.1:8000/overlay-command?command=showBoss&boss=1_Gladius (test nightlord overlay)

### Boss Strength/Weakness Updating

- List of evergaol bosses: https://eip.gg/nightreign/guides/evergaol-bosses/
- List of field bosses: https://eip.gg/nightreign/guides/field-bosses/
- List of bosses (fextra): https://eldenringnightreign.wiki.fextralife.com/Bosses
- Example boss link: https://eldenringnightreign.wiki.fextralife.com/Royal+Carian+Knight
- Strengths/weaknesses stored in: field_boss_detector.py

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
