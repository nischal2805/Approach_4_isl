# ISL Backend API

FastAPI backend for Indian Sign Language translation.

## Features

- **Sign-to-Text**: Upload video → Detect ISL signs → Return English text
- **Text-to-Sign**: Input English text → Return ISL animation frames

## Quick Start

### 1. Install Dependencies

```bash
cd isl_backend
pip install -r requirements.txt
```

### 2. Run Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access API

- API Root: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

### Sign-to-Text

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sign-to-text/classes` | GET | List all recognizable sign classes |
| `/sign-to-text/video` | POST | Process video file |
| `/sign-to-text/frames` | POST | Process batch of frame images |

### Text-to-Sign

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/text-to-sign/words` | GET | List all available sign words |
| `/text-to-sign/translate?text=...` | GET | Get frame paths for text |
| `/text-to-sign/animation?text=...` | GET | Get base64-encoded frames |
| `/text-to-sign/frames/{word}/{file}` | GET | Get specific frame image |

## Flutter Integration

### Sign-to-Text (Video Upload)

```dart
import 'package:http/http.dart' as http;

Future<String> recognizeSigns(File videoFile) async {
  final uri = Uri.parse('http://YOUR_IP:8000/sign-to-text/video');
  final request = http.MultipartRequest('POST', uri);
  request.files.add(await http.MultipartFile.fromPath('video', videoFile.path));
  
  final response = await request.send();
  final body = await response.stream.bytesToString();
  return body; // JSON with detected signs
}
```

### Text-to-Sign (Get Animation)

```dart
Future<List<String>> getSignFrames(String text) async {
  final uri = Uri.parse('http://YOUR_IP:8000/text-to-sign/translate?text=$text');
  final response = await http.get(uri);
  final data = jsonDecode(response.body);
  
  // Returns frame paths for each word
  return data['words'];
}
```

## Architecture

```
isl_backend/
├── main.py              # FastAPI app entry point
├── config.py            # Configuration and paths
├── requirements.txt     # Python dependencies
├── routes/
│   ├── sign_to_text.py  # Sign→Text API routes
│   └── text_to_sign.py  # Text→Sign API routes
└── services/
    ├── sign_to_text.py  # MediaPipe + Model inference
    └── text_to_sign.py  # Frame lookup and serving
```

## Notes

- Sign-to-Text uses the trained model from `isl_recognition/models/`
- Text-to-Sign uses pre-recorded frames from `isl_translation/data/`
- Both services are loaded on startup for fast response times
