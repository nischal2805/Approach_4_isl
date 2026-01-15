import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';

/// Service for communicating with the ISL Translation API
class TranslationService {
  late final Dio _dio;
  
  // TODO: Update with your actual API endpoint
  static const String baseUrl = 'http://localhost:8000';
  // For production: 'https://your-api-gateway.amazonaws.com/prod'
  
  TranslationService() {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 30),
      receiveTimeout: const Duration(seconds: 60),
      headers: {
        'Content-Type': 'application/json',
      },
    ));
  }
  
  /// Update the API base URL
  void setBaseUrl(String url) {
    _dio.options.baseUrl = url;
  }
  
  /// Check if the API is healthy
  Future<bool> healthCheck() async {
    try {
      final response = await _dio.get('/health');
      return response.statusCode == 200 && 
             response.data['model_loaded'] == true;
    } catch (e) {
      return false;
    }
  }
  
  /// Translate a video file
  Future<TranslationResult> translateVideo(File videoFile) async {
    try {
      final formData = FormData.fromMap({
        'video': await MultipartFile.fromFile(
          videoFile.path,
          filename: 'video.mp4',
        ),
      });
      
      final response = await _dio.post('/translate', data: formData);
      
      return TranslationResult(
        translation: response.data['translation'] ?? '',
        confidence: (response.data['confidence'] ?? 0.0).toDouble(),
        processingTimeMs: (response.data['processing_time_ms'] ?? 0.0).toDouble(),
        success: true,
      );
    } on DioException catch (e) {
      return TranslationResult(
        translation: '',
        confidence: 0.0,
        processingTimeMs: 0.0,
        success: false,
        error: _parseError(e),
      );
    } catch (e) {
      return TranslationResult(
        translation: '',
        confidence: 0.0,
        processingTimeMs: 0.0,
        success: false,
        error: 'Unexpected error: $e',
      );
    }
  }
  
  /// Translate video from bytes (base64 encoded)
  Future<TranslationResult> translateVideoBytes(Uint8List videoBytes) async {
    try {
      final base64Video = base64Encode(videoBytes);
      
      final response = await _dio.post('/translate_base64', data: {
        'video': base64Video,
      });
      
      return TranslationResult(
        translation: response.data['translation'] ?? '',
        confidence: (response.data['confidence'] ?? 0.0).toDouble(),
        processingTimeMs: (response.data['processing_time_ms'] ?? 0.0).toDouble(),
        success: true,
      );
    } on DioException catch (e) {
      return TranslationResult(
        translation: '',
        confidence: 0.0,
        processingTimeMs: 0.0,
        success: false,
        error: _parseError(e),
      );
    } catch (e) {
      return TranslationResult(
        translation: '',
        confidence: 0.0,
        processingTimeMs: 0.0,
        success: false,
        error: 'Unexpected error: $e',
      );
    }
  }
  
  String _parseError(DioException e) {
    if (e.response != null) {
      final data = e.response?.data;
      if (data is Map && data.containsKey('detail')) {
        return data['detail'].toString();
      }
      return 'Server error: ${e.response?.statusCode}';
    }
    if (e.type == DioExceptionType.connectionTimeout) {
      return 'Connection timeout. Check your internet.';
    }
    if (e.type == DioExceptionType.connectionError) {
      return 'Cannot connect to server.';
    }
    return 'Network error: ${e.message}';
  }
}

/// Result from translation API
class TranslationResult {
  final String translation;
  final double confidence;
  final double processingTimeMs;
  final bool success;
  final String? error;
  
  TranslationResult({
    required this.translation,
    required this.confidence,
    required this.processingTimeMs,
    required this.success,
    this.error,
  });
  
  @override
  String toString() {
    if (success) {
      return 'Translation: "$translation" (${confidence * 100}% confidence, ${processingTimeMs.toStringAsFixed(0)}ms)';
    }
    return 'Error: $error';
  }
}
