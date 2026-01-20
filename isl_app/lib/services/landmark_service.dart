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
      if (kDebugMode) debugPrint('LandmarkService initialized: $_isInitialized');
      return _isInitialized;
    } catch (e) {
      if (kDebugMode) debugPrint('LandmarkService initialization error: $e');
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
      if (kDebugMode) debugPrint('LandmarkService: Not initialized, attempting init...');
      final success = await initialize();
      if (!success) {
        if (kDebugMode) debugPrint('LandmarkService: Init failed!');
        return [];
      }
    }

    try {
      // Convert CameraImage to bytes for native processing
      final bytes = _convertCameraImageToBytes(image);
      if (bytes == null) {
        if (kDebugMode) debugPrint('LandmarkService: Failed to convert camera image');
        return [];
      }
      
      // Calculate rotation for native side
      int rotation = _calculateRotation();
      
      if (kDebugMode) debugPrint('LandmarkService: Calling native with ${bytes.length} bytes, ${image.width}x${image.height}, rot=$rotation');
      
      // Call native MediaPipe via platform channel
      final result = await _channel.invokeMethod<List<dynamic>>('processFrame', {
        'bytes': bytes,
        'width': image.width,
        'height': image.height,
        'rotation': rotation,
      });
      
      if (result == null) {
        if (kDebugMode) debugPrint('LandmarkService: Native returned null');
        return [];
      }
      
      if (result.isEmpty) {
        if (kDebugMode) debugPrint('LandmarkService: Native returned empty list');
        return [];
      }
      
      // Convert to List<double>
      final landmarks = result.map((e) => (e as num).toDouble()).toList();
      
      // Validate we got the expected 225 features
      if (landmarks.length != 225) {
        if (kDebugMode) debugPrint('LandmarkService: Got ${landmarks.length} landmarks, expected 225');
        return [];
      }
      
      // Check if any meaningful data (not all zeros)
      final nonZeroCount = landmarks.where((v) => v != 0.0).length;
      if (nonZeroCount == 0) {
        if (kDebugMode) debugPrint('LandmarkService: All zeros (no pose detected)');
        return [];
      }
      
      // =================================================================
      // CRITICAL FIX: Normalize Z-coordinates to match Python MediaPipe
      // =================================================================
      // MediaPipe Tasks Vision (Kotlin) outputs z-values ~4-5x larger than
      // Python MediaPipe Holistic used for training:
      //   - Python training z: mean ≈ -0.4, range roughly [-0.6, -0.2]
      //   - Kotlin runtime z:  mean ≈ -1.5 to -2.0, range [-3, 0]
      // 
      // Scale factor: Python z ≈ Kotlin z * 0.25
      // This is applied to every z-coordinate (indices 2, 5, 8, 11, ...)
      const double zScaleFactor = 0.25;
      for (int i = 2; i < landmarks.length; i += 3) {
        landmarks[i] *= zScaleFactor;
      }
      
      // DEBUG: Check landmark value ranges after z-normalization
      if (kDebugMode) {
        final minVal = landmarks.reduce((a, b) => a < b ? a : b);
        final maxVal = landmarks.reduce((a, b) => a > b ? a : b);
        // Sample some values: first pose x,y,z and first hand x,y,z
        debugPrint('LandmarkService: Normalized landmarks range: [${minVal.toStringAsFixed(3)}, ${maxVal.toStringAsFixed(3)}]');
        debugPrint('LandmarkService: Sample pose[0]: x=${landmarks[0].toStringAsFixed(3)}, y=${landmarks[1].toStringAsFixed(3)}, z=${landmarks[2].toStringAsFixed(3)}');
        if (landmarks.length > 102) {
          // First left hand landmark (index 99 = 33*3)
          debugPrint('LandmarkService: Sample hand[0]: x=${landmarks[99].toStringAsFixed(3)}, y=${landmarks[100].toStringAsFixed(3)}, z=${landmarks[101].toStringAsFixed(3)}');
        }
      }
      
      if (kDebugMode) debugPrint('LandmarkService: SUCCESS - ${landmarks.length} landmarks, $nonZeroCount non-zero');
      return landmarks;
    } catch (e) {
      if (kDebugMode) debugPrint('LandmarkService: Error: $e');
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
      
      if (kDebugMode) debugPrint('Unsupported image format: ${image.format.group}');
      return null;
    } catch (e) {
      if (kDebugMode) debugPrint('Error converting image: $e');
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
        if (kDebugMode) debugPrint('LandmarkService disposed');
      } catch (e) {
        if (kDebugMode) debugPrint('Error disposing LandmarkService: $e');
      }
    }
  }
}
