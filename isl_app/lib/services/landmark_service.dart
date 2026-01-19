import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Service to extract landmarks from camera frames using native MediaPipe
/// Communicates with Kotlin via Platform Channel for 75 keypoints:
/// 33 pose + 21 left hand + 21 right hand = 225 features (75 * 3)
class LandmarkService {
  static const _channel = MethodChannel('com.example.isl_translator/landmarks');
  
  bool _isInitialized = false;
  int _sensorOrientation = 0;
  
  /// Initialize the native MediaPipe landmark extractor
  Future<bool> initialize() async {
    if (_isInitialized) return true;
    
    try {
      final result = await _channel.invokeMethod<bool>('initialize');
      _isInitialized = result ?? false;
      debugPrint('LandmarkService initialized: $_isInitialized');
      return _isInitialized;
    } catch (e) {
      debugPrint('LandmarkService initialization error: $e');
      return false;
    }
  }
  
  /// Set sensor orientation from camera controller
  void setSensorOrientation(int orientation) {
    _sensorOrientation = orientation;
  }

  /// Extract 225-dim landmark vector from a camera image
  /// Returns List<double> of size 225 (75 landmarks * 3 coords)
  /// Or empty list if no pose detected
  Future<List<double>> extractLandmarks(CameraImage image) async {
    if (!_isInitialized) {
      // Try to initialize if not done
      final success = await initialize();
      if (!success) return [];
    }

    try {
      // Convert CameraImage to bytes for native processing
      final bytes = _convertCameraImageToBytes(image);
      if (bytes == null) {
        debugPrint('Failed to convert camera image to bytes');
        return [];
      }
      
      // Calculate rotation for native side
      int rotation = _calculateRotation();
      
      // Call native MediaPipe via platform channel
      final result = await _channel.invokeMethod<List<dynamic>>('processFrame', {
        'bytes': bytes,
        'width': image.width,
        'height': image.height,
        'rotation': rotation,
      });
      
      if (result == null || result.isEmpty) {
        return [];
      }
      
      // Convert to List<double>
      final landmarks = result.map((e) => (e as num).toDouble()).toList();
      
      // Validate we got the expected 225 features
      if (landmarks.length != 225) {
        debugPrint('Unexpected landmark count: ${landmarks.length}, expected 225');
        return [];
      }
      
      // Check if any meaningful data (not all zeros)
      final hasData = landmarks.any((v) => v != 0.0);
      if (!hasData) {
        return [];
      }
      
      return landmarks;
    } catch (e) {
      debugPrint('Error extracting landmarks: $e');
      return [];
    }
  }

  /// Convert CameraImage to NV21 bytes for native processing
  Uint8List? _convertCameraImageToBytes(CameraImage image) {
    try {
      // For YUV420 format (Android camera default)
      if (image.format.group == ImageFormatGroup.yuv420) {
        return _yuv420ToNv21(image);
      }
      
      // For NV21 format (some Android devices)
      if (image.format.group == ImageFormatGroup.nv21) {
        return image.planes[0].bytes;
      }
      
      debugPrint('Unsupported image format: ${image.format.group}');
      return null;
    } catch (e) {
      debugPrint('Error converting image: $e');
      return null;
    }
  }
  
  /// Convert YUV420 to NV21 format for MediaPipe
  Uint8List _yuv420ToNv21(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    
    // NV21 format: Y plane followed by interleaved VU
    final int ySize = width * height;
    final int uvSize = (width * height) ~/ 2;
    
    final nv21 = Uint8List(ySize + uvSize);
    
    // Copy Y plane
    int yIndex = 0;
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        nv21[yIndex++] = yPlane.bytes[row * yPlane.bytesPerRow + col];
      }
    }
    
    // Interleave V and U planes (NV21 is VU, not UV)
    int uvIndex = ySize;
    final int uvWidth = width ~/ 2;
    final int uvHeight = height ~/ 2;
    
    for (int row = 0; row < uvHeight; row++) {
      for (int col = 0; col < uvWidth; col++) {
        final int vIdx = row * vPlane.bytesPerRow + col * (vPlane.bytesPerPixel ?? 1);
        final int uIdx = row * uPlane.bytesPerRow + col * (uPlane.bytesPerPixel ?? 1);
        
        nv21[uvIndex++] = vPlane.bytes[vIdx];  // V first (NV21)
        nv21[uvIndex++] = uPlane.bytes[uIdx];  // then U
      }
    }
    
    return nv21;
  }
  
  /// Calculate rotation based on sensor orientation
  int _calculateRotation() {
    // Front camera typically needs 270 degree rotation
    // Adjust based on your device testing
    switch (_sensorOrientation) {
      case 0:
        return 0;
      case 90:
        return 90;
      case 180:
        return 180;
      case 270:
        return 270;
      default:
        return 0;
    }
  }

  /// Release resources
  Future<void> dispose() async {
    if (_isInitialized) {
      try {
        await _channel.invokeMethod('dispose');
        _isInitialized = false;
        debugPrint('LandmarkService disposed');
      } catch (e) {
        debugPrint('Error disposing LandmarkService: $e');
      }
    }
  }
}
