import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Model for a single frame of landmark data
class SignFrame {
  final double timestamp;
  final List<List<double>>? pose;      // 33 landmarks
  final List<List<double>>? leftHand;  // 21 landmarks
  final List<List<double>>? rightHand; // 21 landmarks

  SignFrame({
    required this.timestamp,
    this.pose,
    this.leftHand,
    this.rightHand,
  });

  factory SignFrame.fromJson(Map<String, dynamic> json) {
    return SignFrame(
      timestamp: (json['timestamp'] as num).toDouble(),
      pose: json['pose'] != null
          ? (json['pose'] as List).map((p) => List<double>.from(p)).toList()
          : null,
      leftHand: json['left_hand'] != null
          ? (json['left_hand'] as List).map((p) => List<double>.from(p)).toList()
          : null,
      rightHand: json['right_hand'] != null
          ? (json['right_hand'] as List).map((p) => List<double>.from(p)).toList()
          : null,
    );
  }
}

/// Model for a complete sign animation
class SignData {
  final String label;
  final String type; // "word", "letter", "number"
  final double fps;
  final int frameCount;
  final List<SignFrame> frames;
  final bool isPause;
  final double pauseDuration;

  SignData({
    required this.label,
    required this.type,
    required this.fps,
    required this.frameCount,
    required this.frames,
    this.isPause = false,
    this.pauseDuration = 0.0,
  });

  factory SignData.fromJson(Map<String, dynamic> json) {
    return SignData(
      label: json['label'] as String,
      type: json['type'] as String,
      fps: (json['fps'] as num).toDouble(),
      frameCount: json['frame_count'] as int,
      frames: (json['frames'] as List)
          .map((f) => SignFrame.fromJson(f))
          .toList(),
    );
  }

  /// Create a pause placeholder
  factory SignData.pause({double duration = 0.3}) {
    return SignData(
      label: 'PAUSE',
      type: 'pause',
      fps: 30,
      frameCount: (30 * duration).round(),
      frames: [],
      isPause: true,
      pauseDuration: duration,
    );
  }

  /// Create an unknown word placeholder
  factory SignData.unknown(String word) {
    return SignData(
      label: word,
      type: 'unknown',
      fps: 30,
      frameCount: 30,
      frames: [],
    );
  }
}

/// Service to process text input and generate sign sequences
class TextToSignService {
  Map<String, String> _wordDictionary = {};
  Map<String, String> _letterDictionary = {};
  Map<String, String> _numberDictionary = {};
  bool _isLoaded = false;

  /// Initialize service by loading available signs from index
  Future<void> initialize() async {
    try {
      // Load word index
      await _loadIndex('assets/signs/words/index.json', _wordDictionary);
      
      // Load letter index (fingerspelling)
      await _loadIndex('assets/signs/letters/index.json', _letterDictionary);
      
      // Load number index
      await _loadIndex('assets/signs/numbers/index.json', _numberDictionary);
      
      _isLoaded = true;
      if (kDebugMode) {
        debugPrint('TextToSignService loaded: ${_wordDictionary.length} words, '
            '${_letterDictionary.length} letters, ${_numberDictionary.length} numbers');
      }
    } catch (e) {
      if (kDebugMode) debugPrint('Error loading TextToSignService: $e');
    }
  }

  Future<void> _loadIndex(String path, Map<String, String> dictionary) async {
    try {
      final jsonStr = await rootBundle.loadString(path);
      final index = jsonDecode(jsonStr);
      
      for (var sign in index['signs']) {
        String label = (sign['label'] as String).toLowerCase();
        String file = sign['file'] as String;
        dictionary[label] = path.replaceAll('index.json', file);
      }
    } catch (e) {
      if (kDebugMode) debugPrint('Index not found: $path');
    }
  }

  /// Process input text and return list of sign file paths
  List<SignQueueItem> processText(String input) {
    List<SignQueueItem> signQueue = [];
    
    if (input.trim().isEmpty) return signQueue;
    
    // Normalize input
    String normalized = input.toLowerCase().trim();
    
    // Split into words
    List<String> words = normalized.split(RegExp(r'\s+'));
    
    for (int i = 0; i < words.length; i++) {
      String word = words[i];
      
      // Remove punctuation
      word = word.replaceAll(RegExp(r'[^\w\d]'), '');
      
      if (word.isEmpty) continue;
      
      // Check if word exists in dictionary
      if (_wordDictionary.containsKey(word)) {
        signQueue.add(SignQueueItem(
          type: SignType.word,
          label: word,
          path: _wordDictionary[word]!,
        ));
      } else {
        // Fingerspell the unknown word
        for (int j = 0; j < word.length; j++) {
          String char = word[j].toUpperCase();
          
          if (RegExp(r'[A-Z]').hasMatch(char)) {
            if (_letterDictionary.containsKey(char.toLowerCase())) {
              signQueue.add(SignQueueItem(
                type: SignType.letter,
                label: char,
                path: _letterDictionary[char.toLowerCase()]!,
              ));
            } else {
              // Letter not available
              signQueue.add(SignQueueItem(
                type: SignType.unknown,
                label: char,
                path: '',
              ));
            }
          } else if (RegExp(r'[0-9]').hasMatch(char)) {
            if (_numberDictionary.containsKey(char)) {
              signQueue.add(SignQueueItem(
                type: SignType.number,
                label: char,
                path: _numberDictionary[char]!,
              ));
            } else {
              signQueue.add(SignQueueItem(
                type: SignType.unknown,
                label: char,
                path: '',
              ));
            }
          }
        }
      }
      
      // Add pause between words
      if (i < words.length - 1) {
        signQueue.add(SignQueueItem(
          type: SignType.pause,
          label: 'PAUSE',
          path: '',
        ));
      }
    }
    
    return signQueue;
  }

  /// Load sign data from path
  Future<SignData> loadSign(SignQueueItem item) async {
    if (item.type == SignType.pause) {
      return SignData.pause(duration: 0.3);
    }
    
    if (item.type == SignType.unknown || item.path.isEmpty) {
      return SignData.unknown(item.label);
    }
    
    try {
      final jsonStr = await rootBundle.loadString(item.path);
      return SignData.fromJson(jsonDecode(jsonStr));
    } catch (e) {
      if (kDebugMode) debugPrint('Failed to load sign: ${item.path}');
      return SignData.unknown(item.label);
    }
  }

  bool get isLoaded => _isLoaded;
  int get wordCount => _wordDictionary.length;
  int get letterCount => _letterDictionary.length;
  int get numberCount => _numberDictionary.length;
  
  /// Check if a word is in the dictionary
  bool hasWord(String word) => _wordDictionary.containsKey(word.toLowerCase());
}

enum SignType { word, letter, number, pause, unknown }

class SignQueueItem {
  final SignType type;
  final String label;
  final String path;

  SignQueueItem({
    required this.type,
    required this.label,
    required this.path,
  });
}
