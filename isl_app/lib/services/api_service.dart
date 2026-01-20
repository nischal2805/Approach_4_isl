import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:path_provider/path_provider.dart';
import 'api_config_service.dart';

/// API Service for ISL Backend
/// Handles both Sign-to-Text and Text-to-Sign via HTTP
class ApiService {
  // Use ApiConfigService for dynamic IP configuration
  final ApiConfigService _config = ApiConfigService();
  
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  bool _isConnected = false;
  String? _lastError;

  bool get isConnected => _isConnected;
  String? get lastError => _lastError;
  
  /// Get the base URL from config (dynamic, not hardcoded!)
  String? get baseUrl => _config.baseUrl;
  
  /// Check if API is configured with an IP address
  bool get isConfigured => _config.currentIp != null && _config.currentIp!.isNotEmpty;

  /// Check if backend is available
  Future<bool> checkConnection() async {
    // Make sure config is initialized
    await _config.initialize();
    
    // If no IP configured, try auto-discovery
    if (!isConfigured) {
      if (kDebugMode) debugPrint('API: No IP configured, trying auto-discovery...');
      final discoveredIp = await _config.autoDiscover();
      if (discoveredIp == null) {
        _isConnected = false;
        _lastError = 'Backend not found on network. Make sure backend is running.';
        return false;
      }
      if (kDebugMode) debugPrint('API: Auto-discovered backend at $discoveredIp');
    }
    
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
      ).timeout(const Duration(seconds: 5));
      
      _isConnected = response.statusCode == 200;
      if (_isConnected) {
        final data = jsonDecode(response.body);
        if (kDebugMode) {
          debugPrint('API: Connected - ${data['status']}');
          debugPrint('API: Sign-to-Text classes: ${data['services']['sign_to_text']['classes_count']}');
          debugPrint('API: Text-to-Sign words: ${data['services']['text_to_sign']['words_count']}');
        }
      }
      return _isConnected;
    } catch (e) {
      _lastError = e.toString();
      _isConnected = false;
      if (kDebugMode) debugPrint('API: Connection failed - $e');
      return false;
    }
  }

  // ==================== SIGN TO TEXT ====================

  /// Get list of recognizable sign classes
  Future<List<String>> getSignClasses() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/sign-to-text/classes'),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return List<String>.from(data['classes']);
      }
      return [];
    } catch (e) {
      _lastError = e.toString();
      return [];
    }
  }

  /// Process a video file and get detected signs
  Future<SignToTextResult?> processVideo(File videoFile) async {
    try {
      if (kDebugMode) debugPrint('API: Uploading video ${videoFile.path}');
      
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/sign-to-text/video'),
      );
      
      // Read file bytes and add with explicit content type
      final bytes = await videoFile.readAsBytes();
      if (kDebugMode) debugPrint('API: Read ${bytes.length} bytes from file');
      
      request.files.add(http.MultipartFile.fromBytes(
        'video',
        bytes,
        filename: 'video.mp4',
        contentType: MediaType('video', 'mp4'),
      ));

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 60),
      );
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return SignToTextResult.fromJson(data);
      } else {
        _lastError = 'Server error: ${response.statusCode}';
        if (kDebugMode) debugPrint('API Error: ${response.body}');
        return null;
      }
    } catch (e) {
      _lastError = e.toString();
      if (kDebugMode) debugPrint('API: Video upload failed - $e');
      return null;
    }
  }

  /// Process raw video bytes
  Future<SignToTextResult?> processVideoBytes(Uint8List videoBytes, String filename) async {
    try {
      if (kDebugMode) debugPrint('API: Uploading ${videoBytes.length} bytes');
      
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/sign-to-text/video'),
      );
      
      request.files.add(http.MultipartFile.fromBytes(
        'video',
        videoBytes,
        filename: filename,
        contentType: MediaType('video', 'mp4'),
      ));

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 60),
      );
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return SignToTextResult.fromJson(data);
      } else {
        _lastError = 'Server error: ${response.statusCode}';
        return null;
      }
    } catch (e) {
      _lastError = e.toString();
      return null;
    }
  }

  // ==================== TEXT TO SIGN ====================

  /// Get list of available sign words
  Future<List<String>> getAvailableWords() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/text-to-sign/words'),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return List<String>.from(data['words']);
      }
      return [];
    } catch (e) {
      _lastError = e.toString();
      return [];
    }
  }

  /// Translate text to sign frame paths
  Future<TextToSignResult?> translateText(String text) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/text-to-sign/translate?text=${Uri.encodeComponent(text)}'),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return TextToSignResult.fromJson(data);
      } else {
        _lastError = 'Server error: ${response.statusCode}';
        return null;
      }
    } catch (e) {
      _lastError = e.toString();
      return null;
    }
  }

  /// Get frame image URL
  String getFrameUrl(String framePath) {
    return '$baseUrl/static/frames/$framePath';
  }

  /// Get all frames for a specific word
  Future<List<String>> getWordFrames(String word) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/text-to-sign/word/${Uri.encodeComponent(word)}'),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return List<String>.from(data['frame_paths']);
      }
      return [];
    } catch (e) {
      _lastError = e.toString();
      return [];
    }
  }

  /// Get animation data with base64 encoded frames
  Future<AnimationResult?> getAnimationData(String text) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/text-to-sign/animation?text=${Uri.encodeComponent(text)}'),
      ).timeout(const Duration(seconds: 30));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return AnimationResult.fromJson(data);
      } else {
        _lastError = 'Server error: ${response.statusCode}';
        return null;
      }
    } catch (e) {
      _lastError = e.toString();
      return null;
    }
  }
}

// ==================== DATA MODELS ====================

class SignPrediction {
  final String sign;
  final double confidence;
  final int startFrame;
  final int endFrame;
  final double startTime;
  final double endTime;
  final Map<String, double> topPredictions;

  SignPrediction({
    required this.sign,
    required this.confidence,
    this.startFrame = 0,
    this.endFrame = 0,
    this.startTime = 0.0,
    this.endTime = 0.0,
    this.topPredictions = const {},
  });

  factory SignPrediction.fromJson(Map<String, dynamic> json) {
    return SignPrediction(
      sign: json['sign'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      startFrame: json['start_frame'] as int? ?? 0,
      endFrame: json['end_frame'] as int? ?? 0,
      startTime: (json['start_time'] as num?)?.toDouble() ?? 0.0,
      endTime: (json['end_time'] as num?)?.toDouble() ?? 0.0,
      topPredictions: json['top_predictions'] != null
          ? Map<String, double>.from(
              (json['top_predictions'] as Map).map(
                (k, v) => MapEntry(k.toString(), (v as num).toDouble()),
              ),
            )
          : {},
    );
  }
}

class SignToTextResult {
  final bool success;
  final List<SignPrediction> signs;
  final String sentence;
  final String message;

  SignToTextResult({
    required this.success,
    required this.signs,
    required this.sentence,
    this.message = '',
  });

  factory SignToTextResult.fromJson(Map<String, dynamic> json) {
    return SignToTextResult(
      success: json['success'] as bool,
      signs: (json['signs'] as List)
          .map((s) => SignPrediction.fromJson(s))
          .toList(),
      sentence: json['sentence'] as String,
      message: json['message'] as String? ?? '',
    );
  }
}

class WordSignResult {
  final String word;
  final bool found;
  final int framesCount;
  final List<String> framePaths;

  WordSignResult({
    required this.word,
    required this.found,
    required this.framesCount,
    required this.framePaths,
  });

  factory WordSignResult.fromJson(Map<String, dynamic> json) {
    return WordSignResult(
      word: json['word'] as String,
      found: json['found'] as bool,
      framesCount: json['frames_count'] as int,
      framePaths: List<String>.from(json['frame_paths']),
    );
  }
}

class TextToSignResult {
  final bool success;
  final String originalText;
  final List<WordSignResult> words;
  final int foundCount;
  final int totalCount;

  TextToSignResult({
    required this.success,
    required this.originalText,
    required this.words,
    required this.foundCount,
    required this.totalCount,
  });

  factory TextToSignResult.fromJson(Map<String, dynamic> json) {
    return TextToSignResult(
      success: json['success'] as bool,
      originalText: json['original_text'] as String,
      words: (json['words'] as List)
          .map((w) => WordSignResult.fromJson(w))
          .toList(),
      foundCount: json['found_count'] as int,
      totalCount: json['total_count'] as int,
    );
  }
}

class AnimationFrame {
  final String filename;
  final Uint8List data;

  AnimationFrame({required this.filename, required this.data});

  factory AnimationFrame.fromJson(Map<String, dynamic> json) {
    return AnimationFrame(
      filename: json['filename'] as String,
      data: base64Decode(json['data_base64'] as String),
    );
  }
}

class AnimationWord {
  final String word;
  final bool found;
  final List<AnimationFrame> frames;

  AnimationWord({
    required this.word,
    required this.found,
    required this.frames,
  });

  factory AnimationWord.fromJson(Map<String, dynamic> json) {
    return AnimationWord(
      word: json['word'] as String,
      found: json['found'] as bool,
      frames: (json['frames'] as List)
          .map((f) => AnimationFrame.fromJson(f))
          .toList(),
    );
  }
}

class AnimationResult {
  final bool success;
  final String originalText;
  final List<AnimationWord> animations;

  AnimationResult({
    required this.success,
    required this.originalText,
    required this.animations,
  });

  factory AnimationResult.fromJson(Map<String, dynamic> json) {
    return AnimationResult(
      success: json['success'] as bool,
      originalText: json['original_text'] as String,
      animations: (json['animations'] as List)
          .map((a) => AnimationWord.fromJson(a))
          .toList(),
    );
  }
}
