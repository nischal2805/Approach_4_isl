# Flutter App Integration Guide for ISL Translation Model

## 📌 Model Overview

This document provides everything a Flutter developer needs to integrate the ISL (Indian Sign Language) to English translation model into a mobile application.

### Architecture Summary

```
Video Recording (Flutter App)
       ↓
Raw Video Frames [30 frames, 224x224, RGB]
       ↓
   [API Request to Backend Server]
       ↓
I3D Encoder (frozen, pretrained on Kinetics-400)
       ↓
BiLSTM Temporal Encoder (2 layers, 512 hidden)
       ↓
Projection + Positional Encoding
       ↓
T5-Small Decoder (pretrained)
       ↓
English Text Translation
```

---

## ⚠️ CRITICAL: Model Deployment Options

The model is **~100MB+ in size** and requires **PyTorch + CUDA/CPU runtime**. This means:

| Option | Feasibility | Notes |
|--------|-------------|-------|
| **On-device (Flutter)** | ❌ Not Recommended | Model is too large for mobile, requires Python runtime |
| **Backend Server API** | ✅ Recommended | Deploy FastAPI server, Flutter sends video to API |
| **Edge Device (Jetson, etc.)** | ⚠️ Possible | For kiosk/tablet deployments with dedicated hardware |

**Recommended Architecture:**
```
┌─────────────────┐                    ┌─────────────────────┐
│   Flutter App   │  ────HTTP POST───▶ │   FastAPI Server    │
│  (Video Record) │                    │ (GPU/CPU Inference) │
│                 │  ◀───JSON Response │                     │
└─────────────────┘                    └─────────────────────┘
```

---

## 📋 Input Requirements

### Video Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Frame Count** | 30 frames | Uniformly sampled from video |
| **Frame Size** | 224 × 224 pixels | Resized with preserved aspect or center crop |
| **Color Format** | RGB | Convert from BGR if using OpenCV |
| **Normalization** | ImageNet | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| **Tensor Shape** | `[1, 3, 30, 224, 224]` | [Batch, Channels, Time, Height, Width] |
| **Data Type** | Float32 | Values normalized to [0, 1] then standardized |

### Recording Guidelines for Users

> [!IMPORTANT]
> **User Recording Instructions:**
> - Record for **2-4 seconds** (minimum 30 frames at 15 FPS)
> - Keep the **signer centered** in the frame
> - Ensure **good lighting** (avoid backlight)
> - Keep **camera steady** (minimal shake)
> - Sign at **normal pace** (not too fast)

---

## 🔌 API Endpoints

The backend server (FastAPI) provides these endpoints:

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 2. Translate Video (File Upload)
```http
POST /translate
Content-Type: multipart/form-data

video: <video file (mp4, mov, avi, webm)>
```
**Response:**
```json
{
  "translation": "hello how are you",
  "confidence": 0.85,
  "processing_time_ms": 1523.45
}
```

### 3. Translate Video (Base64)
```http
POST /translate_base64
Content-Type: application/json

{
  "video": "<base64 encoded video bytes>"
}
```
**Response:** Same as above

---

## 📱 Flutter Implementation

### Dependencies (pubspec.yaml)

```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.10.5+9       # Camera access
  video_player: ^2.8.2    # Video playback
  http: ^1.1.0            # HTTP requests
  path_provider: ^2.1.1   # File storage
  permission_handler: ^11.0.1  # Permissions
```

### Flutter Code Example

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:camera/camera.dart';

class ISLTranslationService {
  // Backend server URL (replace with your deployed server)
  static const String API_BASE_URL = 'http://YOUR_SERVER_IP:8000';
  
  /// Translate a video file to English text
  static Future<TranslationResult> translateVideo(File videoFile) async {
    final uri = Uri.parse('$API_BASE_URL/translate');
    
    var request = http.MultipartRequest('POST', uri);
    request.files.add(
      await http.MultipartFile.fromPath('video', videoFile.path),
    );
    
    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);
    
    if (response.statusCode == 200) {
      final json = jsonDecode(response.body);
      return TranslationResult(
        translation: json['translation'],
        confidence: json['confidence'],
        processingTimeMs: json['processing_time_ms'],
      );
    } else {
      throw Exception('Translation failed: ${response.body}');
    }
  }
  
  /// Translate video as base64 (alternative method)
  static Future<TranslationResult> translateVideoBase64(File videoFile) async {
    final bytes = await videoFile.readAsBytes();
    final base64Video = base64Encode(bytes);
    
    final response = await http.post(
      Uri.parse('$API_BASE_URL/translate_base64'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'video': base64Video}),
    );
    
    if (response.statusCode == 200) {
      final json = jsonDecode(response.body);
      return TranslationResult(
        translation: json['translation'],
        confidence: json['confidence'],
        processingTimeMs: json['processing_time_ms'],
      );
    } else {
      throw Exception('Translation failed: ${response.body}');
    }
  }
  
  /// Check if the backend server is healthy
  static Future<bool> checkHealth() async {
    try {
      final response = await http.get(
        Uri.parse('$API_BASE_URL/health'),
      ).timeout(Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        return json['model_loaded'] == true;
      }
    } catch (e) {
      print('Health check failed: $e');
    }
    return false;
  }
}

class TranslationResult {
  final String translation;
  final double confidence;
  final double processingTimeMs;
  
  TranslationResult({
    required this.translation,
    required this.confidence,
    required this.processingTimeMs,
  });
}
```

### Recording Widget Example

```dart
class VideoRecordingScreen extends StatefulWidget {
  @override
  _VideoRecordingScreenState createState() => _VideoRecordingScreenState();
}

class _VideoRecordingScreenState extends State<VideoRecordingScreen> {
  CameraController? _controller;
  bool _isRecording = false;
  String? _translation;
  bool _isTranslating = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );
    
    _controller = CameraController(
      frontCamera,
      ResolutionPreset.medium,  // 720p is sufficient
      enableAudio: false,       // No audio needed for sign language
    );
    
    await _controller!.initialize();
    setState(() {});
  }

  Future<void> _startRecording() async {
    if (_controller == null || _isRecording) return;
    
    await _controller!.startVideoRecording();
    setState(() => _isRecording = true);
    
    // Auto-stop after 3 seconds
    Future.delayed(Duration(seconds: 3), () {
      if (_isRecording) _stopAndTranslate();
    });
  }

  Future<void> _stopAndTranslate() async {
    if (!_isRecording) return;
    
    final videoFile = await _controller!.stopVideoRecording();
    setState(() {
      _isRecording = false;
      _isTranslating = true;
    });
    
    try {
      final result = await ISLTranslationService.translateVideo(
        File(videoFile.path),
      );
      setState(() {
        _translation = result.translation;
        _isTranslating = false;
      });
    } catch (e) {
      setState(() {
        _translation = 'Error: $e';
        _isTranslating = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    // ... UI implementation
  }
}
```

---

## 🖥️ Backend Server Deployment

### Running Locally

```bash
cd isl_translation

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart

# Set model path (optional, defaults to checkpoints/best_model.pt)
export MODEL_PATH=checkpoints/server_best_model.pt

# Run server
python inference/api.py
# OR
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

### Running with Docker

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn python-multipart mangum

ENV MODEL_PATH=/app/checkpoints/best_model.pt

EXPOSE 8000
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment Options

| Platform | Recommended Tier | Notes |
|----------|-----------------|-------|
| **AWS EC2** | g4dn.xlarge (GPU) | Best for production with GPU |
| **Google Cloud Run** | CPU only | Serverless, cold starts |
| **AWS Lambda** | Via Mangum | Cold start issues, CPU only |
| **Heroku** | Performance-L | CPU only, limited |

---

## 📊 Performance Expectations

| Device | Inference Time | Notes |
|--------|---------------|-------|
| **NVIDIA A100** | ~500ms | Production server |
| **NVIDIA T4** | ~800ms | Cloud GPU (EC2 g4dn) |
| **NVIDIA RTX 3060** | ~600ms | Local development |
| **CPU (8 cores)** | ~3-5 seconds | Fallback option |

---

## 🚫 What the Flutter App Should NOT Do

> [!CAUTION]
> The following should be handled by the backend server, NOT the Flutter app:

1. **Landmark extraction** - Not needed! The model uses raw video (I3D extracts features automatically)
2. **Frame preprocessing** - Server handles resizing, normalization, sampling
3. **Model inference** - Requires PyTorch runtime (not available on mobile)
4. **T5 tokenization** - Handled by the model internally

---

## ✅ What the Flutter App SHOULD Do

1. **Camera access** - Front camera for recording
2. **Video recording** - 2-4 seconds of signing
3. **File compression** - Optional, reduce upload size
4. **HTTP request** - Send video to backend API
5. **Display results** - Show translation to user
6. **Error handling** - Network errors, server errors
7. **Loading states** - Show progress during translation
8. **Offline detection** - Check connectivity before sending

---

## 🔐 Security Considerations

1. **HTTPS** - Use HTTPS in production
2. **API Keys** - Add authentication for rate limiting
3. **File Size Limits** - Limit video uploads (~10MB max)
4. **Input Validation** - Validate video format on server

---

## 📁 File Structure Reference

```
isl_translation/
├── models/
│   ├── i3d_encoder.py      # I3D visual encoder
│   ├── temporal_encoder.py # BiLSTM + projection
│   └── translator.py       # Full model (ISLTranslator)
├── inference/
│   └── api.py              # FastAPI server (deploy this)
├── checkpoints/
│   └── best_model.pt       # Trained model weights
├── configs/
│   └── train_config.yaml   # Model configuration
└── FLUTTER_INTEGRATION.md  # This document
```

---

## 🧪 Testing the Integration

### 1. Test Server Health
```bash
curl http://YOUR_SERVER:8000/health
```

### 2. Test Translation (with sample video)
```bash
curl -X POST "http://YOUR_SERVER:8000/translate" \
  -F "video=@test_video.mp4"
```

### 3. Test from Flutter
```dart
void testIntegration() async {
  final isHealthy = await ISLTranslationService.checkHealth();
  print('Server healthy: $isHealthy');
  
  if (isHealthy) {
    // Record a video and test translation
    // ...
  }
}
```

---

## 📞 Support

For issues or questions about this integration:
- Check server logs: `logs/` directory
- Verify model loading: Check `/health` endpoint
- Test with sample videos from training dataset first

---

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Model:** ISL Translation (I3D + BiLSTM + T5-small)
