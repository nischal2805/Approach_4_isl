import 'dart:io';
import 'dart:typed_data';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

/// Service to extract landmarks from camera frames using Google ML Kit
class LandmarkService {
  late final PoseDetector _poseDetector;
  bool _isProcessing = false;

  LandmarkService() {
    final options = PoseDetectorOptions(
      mode: PoseDetectionMode.stream,
      model: PoseDetectionModel.base,
    );
    _poseDetector = PoseDetector(options: options);
  }

  /// Extract 225-dim landmark vector from a camera image
  /// Returns List<double> of size 225 (75 landmarks * 3 coords)
  /// Or empty list if no pose detected
  Future<List<double>> extractLandmarks(CameraImage image) async {
    if (_isProcessing) return [];
    _isProcessing = true;

    try {
      final inputImage = _convertCameraImage(image);
      if (inputImage == null) return [];

      final poses = await _poseDetector.processImage(inputImage);
      
      if (poses.isEmpty) return [];

      final pose = poses.first;
      
      // Output 225 floats (75 points x 3)
      // ML Kit gives 33 pose points, hands are padded with zeros
      List<double> landmarks = List.filled(225, 0.0);

      // Map 33 Pose landmarks
      pose.landmarks.forEach((type, landmark) {
        int index = type.index; 
        if (index < 33) {
          int offset = index * 3;
          // Normalize to [0, 1] based on image dimensions
          landmarks[offset] = landmark.x;
          landmarks[offset + 1] = landmark.y;
          landmarks[offset + 2] = landmark.z ?? 0.0;
        }
      });
      
      // Hand landmarks (33-74) remain zero - ML Kit Pose doesn't provide detailed hand tracking
      // For better accuracy, consider adding google_mlkit_selfie_segmentation or hand_landmarker
      
      return landmarks;
    } catch (e) {
      debugPrint('Error extracting landmarks: $e');
      return [];
    } finally {
      _isProcessing = false;
    }
  }

  /// Convert CameraImage to InputImage for ML Kit
  InputImage? _convertCameraImage(CameraImage image) {
    try {
      // Get rotation
      final inputImageRotation = InputImageRotation.rotation0deg; // Adjust based on camera orientation
      
      // For YUV420 format (most common on Android)
      if (image.format.group == ImageFormatGroup.yuv420) {
        final WriteBuffer allBytes = WriteBuffer();
        for (final plane in image.planes) {
          allBytes.putUint8List(plane.bytes);
        }
        final bytes = allBytes.done().buffer.asUint8List();

        final inputImageFormat = InputImageFormat.yuv420;

        final planeData = image.planes.map(
          (plane) => InputImagePlaneMetadata(
            bytesPerRow: plane.bytesPerRow,
            height: plane.height ?? image.height,
            width: plane.width ?? image.width,
          ),
        ).toList();

        final inputImageData = InputImageData(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          imageRotation: inputImageRotation,
          inputImageFormat: inputImageFormat,
          planeData: planeData,
        );

        return InputImage.fromBytes(bytes: bytes, inputImageData: inputImageData);
      }
      
      // For BGRA8888 format (iOS)
      if (image.format.group == ImageFormatGroup.bgra8888) {
        final bytes = image.planes.first.bytes;
        
        final inputImageData = InputImageData(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          imageRotation: inputImageRotation,
          inputImageFormat: InputImageFormat.bgra8888,
          planeData: [
            InputImagePlaneMetadata(
              bytesPerRow: image.planes.first.bytesPerRow,
              height: image.height,
              width: image.width,
            )
          ],
        );
        
        return InputImage.fromBytes(bytes: bytes, inputImageData: inputImageData);
      }
      
      debugPrint('Unsupported image format: ${image.format.group}');
      return null;
    } catch (e) {
      debugPrint('Error converting camera image: $e');
      return null;
    }
  }
  
  void dispose() {
    _poseDetector.close();
  }
}
