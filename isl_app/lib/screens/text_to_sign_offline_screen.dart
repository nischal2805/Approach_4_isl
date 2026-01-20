import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

/// Text to Sign Screen - OFFLINE (Local JSON Assets)
/// Converts English text to ISL sign animations using landmark stick figures
class TextToSignOfflineScreen extends StatefulWidget {
  const TextToSignOfflineScreen({super.key});

  @override
  State<TextToSignOfflineScreen> createState() => _TextToSignOfflineScreenState();
}

class _TextToSignOfflineScreenState extends State<TextToSignOfflineScreen> {
  final TextEditingController _textController = TextEditingController();

  bool _isLoading = false;
  String? _errorMessage;

  // Available words from assets
  List<String> _availableWords = [];
  
  // Animation data
  List<SignAnimationData> _signAnimations = [];
  List<LandmarkFrame> _allFrames = [];
  List<String> _wordLabels = [];
  int _currentFrameIndex = 0;
  Timer? _animationTimer;
  bool _isPlaying = false;

  // Playback speed (ms per frame)
  int _frameDelay = 100;

  @override
  void initState() {
    super.initState();
    _loadAvailableWords();
  }

  @override
  void dispose() {
    _animationTimer?.cancel();
    _textController.dispose();
    super.dispose();
  }

  /// Load list of available words from assets
  Future<void> _loadAvailableWords() async {
    try {
      // Load the asset manifest
      final manifestJson = await rootBundle.loadString('AssetManifest.json');
      final manifest = jsonDecode(manifestJson) as Map<String, dynamic>;
      
      // Find all JSON files in assets/signs/words/
      final words = <String>[];
      for (final key in manifest.keys) {
        if (key.startsWith('assets/signs/words/') && key.endsWith('.json')) {
          // Extract word name from path
          final filename = key.split('/').last;
          final word = filename.replaceAll('.json', '');
          words.add(word);
        }
      }
      
      words.sort();
      
      if (mounted) {
        setState(() {
          _availableWords = words;
        });
      }
      
      if (kDebugMode) {
        debugPrint('TextToSign Offline: ${words.length} words available');
      }
    } catch (e) {
      if (kDebugMode) debugPrint('Error loading word list: $e');
    }
  }

  /// Load animation data for a specific word
  Future<SignAnimationData?> _loadWordAnimation(String word) async {
    try {
      final jsonPath = 'assets/signs/words/$word.json';
      final jsonStr = await rootBundle.loadString(jsonPath);
      final jsonData = jsonDecode(jsonStr) as List<dynamic>;
      
      final frames = <LandmarkFrame>[];
      for (final frameData in jsonData) {
        frames.add(LandmarkFrame.fromJson(frameData as Map<String, dynamic>));
      }
      
      return SignAnimationData(word: word, frames: frames, found: true);
    } catch (e) {
      if (kDebugMode) debugPrint('Word "$word" not found: $e');
      return SignAnimationData(word: word, frames: [], found: false);
    }
  }

  /// Translate text to sign animations
  Future<void> _translateText() async {
    final text = _textController.text.trim().toLowerCase();
    if (text.isEmpty) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _stopAnimation();
      _allFrames.clear();
      _wordLabels.clear();
      _signAnimations.clear();
    });

    try {
      // Split text into words
      final words = text.split(RegExp(r'\s+'));
      final animations = <SignAnimationData>[];
      
      for (final word in words) {
        // Clean the word
        final cleanWord = word.replaceAll(RegExp(r'[^\w]'), '');
        if (cleanWord.isEmpty) continue;
        
        // Try different formats
        SignAnimationData? anim;
        
        // Try exact match
        anim = await _loadWordAnimation(cleanWord);
        if (anim == null || !anim.found) {
          // Try with underscores for multi-word phrases
          anim = await _loadWordAnimation(cleanWord.replaceAll(' ', '_'));
        }
        
        animations.add(anim ?? SignAnimationData(word: cleanWord, frames: [], found: false));
      }

      // Flatten all frames with word labels
      final allFrames = <LandmarkFrame>[];
      final labels = <String>[];
      
      for (final anim in animations) {
        if (anim.found) {
          for (final frame in anim.frames) {
            allFrames.add(frame);
            labels.add(anim.word);
          }
        }
      }

      setState(() {
        _signAnimations = animations;
        _allFrames = allFrames;
        _wordLabels = labels;
        _currentFrameIndex = 0;
        _isLoading = false;
      });

      if (_allFrames.isNotEmpty) {
        _startAnimation();
      } else {
        setState(() {
          _errorMessage = 'No signs found for the entered text';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  void _startAnimation() {
    if (_allFrames.isEmpty) return;
    
    _animationTimer?.cancel();
    _isPlaying = true;
    
    _animationTimer = Timer.periodic(Duration(milliseconds: _frameDelay), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      setState(() {
        _currentFrameIndex = (_currentFrameIndex + 1) % _allFrames.length;
      });
    });
    
    setState(() {});
  }

  void _stopAnimation() {
    _animationTimer?.cancel();
    _isPlaying = false;
    setState(() {});
  }

  void _togglePlayPause() {
    if (_isPlaying) {
      _stopAnimation();
    } else {
      _startAnimation();
    }
  }

  void _previousFrame() {
    if (_allFrames.isEmpty) return;
    _stopAnimation();
    setState(() {
      _currentFrameIndex = (_currentFrameIndex - 1 + _allFrames.length) % _allFrames.length;
    });
  }

  void _nextFrame() {
    if (_allFrames.isEmpty) return;
    _stopAnimation();
    setState(() {
      _currentFrameIndex = (_currentFrameIndex + 1) % _allFrames.length;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'Text to Sign (Offline)',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Row(
              children: [
                const Icon(Icons.offline_bolt, color: Colors.amber, size: 20),
                const SizedBox(width: 4),
                Text(
                  '${_availableWords.length} words',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // Text input section
          _buildInputSection(),

          // Animation display
          Expanded(
            child: _buildAnimationSection(),
          ),

          // Playback controls
          if (_allFrames.isNotEmpty) _buildPlaybackControls(),

          // Word chips showing available/found words
          _buildWordStatus(),
        ],
      ),
    );
  }

  Widget _buildInputSection() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _textController,
              style: const TextStyle(color: Colors.white),
              decoration: const InputDecoration(
                hintText: 'Enter text to translate...',
                hintStyle: TextStyle(color: Colors.white38),
                border: InputBorder.none,
              ),
              onSubmitted: (_) => _translateText(),
            ),
          ),
          IconButton(
            icon: _isLoading
                ? const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                  )
                : const Icon(Icons.send, color: Colors.white),
            onPressed: _isLoading ? null : _translateText,
          ),
        ],
      ),
    );
  }

  Widget _buildAnimationSection() {
    if (_errorMessage != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, color: Colors.orange, size: 48),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              style: const TextStyle(color: Colors.orange),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            TextButton(
              onPressed: () {
                setState(() => _errorMessage = null);
              },
              child: const Text('Dismiss'),
            ),
          ],
        ),
      );
    }

    if (_isLoading) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SpinKitWave(color: Colors.white, size: 40),
            SizedBox(height: 16),
            Text(
              'Loading sign animation...',
              style: TextStyle(color: Colors.white70),
            ),
          ],
        ),
      );
    }

    if (_allFrames.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.sign_language, size: 80, color: Colors.white.withOpacity(0.3)),
            const SizedBox(height: 16),
            Text(
              'Enter text above to see sign animation',
              style: TextStyle(color: Colors.white.withOpacity(0.5)),
            ),
            if (_availableWords.isNotEmpty) ...[
              const SizedBox(height: 24),
              Text(
                'Try: ${_availableWords.take(8).join(", ")}...',
                style: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 12),
                textAlign: TextAlign.center,
              ),
            ],
          ],
        ),
      );
    }

    // Show current frame as stick figure
    final currentFrame = _allFrames[_currentFrameIndex];
    final currentWord = _currentFrameIndex < _wordLabels.length 
        ? _wordLabels[_currentFrameIndex] 
        : '';

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white24),
        color: Colors.black26,
      ),
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Stick figure rendering
          CustomPaint(
            painter: LandmarkPainter(frame: currentFrame),
          ),

          // Word label overlay
          Positioned(
            top: 16,
            left: 16,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                currentWord.toUpperCase().replaceAll('_', ' '),
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ),
          ),

          // Frame counter
          Positioned(
            bottom: 16,
            right: 16,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                '${_currentFrameIndex + 1} / ${_allFrames.length}',
                style: const TextStyle(color: Colors.white70, fontSize: 14),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPlaybackControls() {
    return Container(
      margin: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Speed slider
          Row(
            children: [
              const Icon(Icons.slow_motion_video, color: Colors.white54, size: 20),
              Expanded(
                child: Slider(
                  value: _frameDelay.toDouble(),
                  min: 50,
                  max: 500,
                  divisions: 9,
                  label: '${_frameDelay}ms',
                  onChanged: (value) {
                    setState(() => _frameDelay = value.round());
                    if (_isPlaying) {
                      _stopAnimation();
                      _startAnimation();
                    }
                  },
                ),
              ),
              const Icon(Icons.speed, color: Colors.white54, size: 20),
            ],
          ),
          
          // Playback buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
                icon: const Icon(Icons.skip_previous, color: Colors.white, size: 32),
                onPressed: _previousFrame,
              ),
              const SizedBox(width: 16),
              Container(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white.withOpacity(0.2),
                ),
                child: IconButton(
                  icon: Icon(
                    _isPlaying ? Icons.pause : Icons.play_arrow,
                    color: Colors.white,
                    size: 40,
                  ),
                  onPressed: _togglePlayPause,
                ),
              ),
              const SizedBox(width: 16),
              IconButton(
                icon: const Icon(Icons.skip_next, color: Colors.white, size: 32),
                onPressed: _nextFrame,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildWordStatus() {
    if (_signAnimations.isEmpty) return const SizedBox.shrink();
    
    return Container(
      margin: const EdgeInsets.only(bottom: 16, left: 16, right: 16),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        alignment: WrapAlignment.center,
        children: _signAnimations.map((anim) {
          return Chip(
            label: Text(
              anim.word.replaceAll('_', ' '),
              style: TextStyle(
                color: anim.found ? Colors.white : Colors.white54,
              ),
            ),
            backgroundColor: anim.found 
                ? Colors.green.withOpacity(0.3) 
                : Colors.red.withOpacity(0.3),
            avatar: Icon(
              anim.found ? Icons.check : Icons.close,
              size: 16,
              color: anim.found ? Colors.green : Colors.red,
            ),
          );
        }).toList(),
      ),
    );
  }
}

// ==================== DATA MODELS ====================

class SignAnimationData {
  final String word;
  final List<LandmarkFrame> frames;
  final bool found;

  SignAnimationData({
    required this.word,
    required this.frames,
    required this.found,
  });
}

class LandmarkFrame {
  final double timestamp;
  final List<List<double>> pose;      // 33 landmarks, each [x, y, z]
  final List<List<double>> leftHand;  // 21 landmarks, each [x, y, z]
  final List<List<double>> rightHand; // 21 landmarks, each [x, y, z]

  LandmarkFrame({
    required this.timestamp,
    required this.pose,
    required this.leftHand,
    required this.rightHand,
  });

  factory LandmarkFrame.fromJson(Map<String, dynamic> json) {
    return LandmarkFrame(
      timestamp: (json['timestamp'] as num).toDouble(),
      pose: _parseLandmarks(json['pose']),
      leftHand: _parseLandmarks(json['left_hand']),
      rightHand: _parseLandmarks(json['right_hand']),
    );
  }

  static List<List<double>> _parseLandmarks(List<dynamic>? data) {
    if (data == null) return [];
    return data.map((point) {
      if (point is List) {
        return point.map((v) => (v as num).toDouble()).toList();
      }
      return <double>[0, 0, 0];
    }).toList();
  }
}

// ==================== STICK FIGURE PAINTER ====================

class LandmarkPainter extends CustomPainter {
  final LandmarkFrame frame;

  LandmarkPainter({required this.frame});

  // MediaPipe pose connections (simplified for major body parts)
  static const List<List<int>> poseConnections = [
    // Face/head
    [0, 1], [1, 2], [2, 3], [3, 7],  // Left eye
    [0, 4], [4, 5], [5, 6], [6, 8],  // Right eye
    [9, 10],  // Mouth
    // Torso
    [11, 12],  // Shoulders
    [11, 23], [12, 24],  // Torso sides
    [23, 24],  // Hips
    // Left arm
    [11, 13], [13, 15],  // Shoulder to wrist
    // Right arm
    [12, 14], [14, 16],  // Shoulder to wrist
    // Left leg
    [23, 25], [25, 27], [27, 29], [27, 31],
    // Right leg
    [24, 26], [26, 28], [28, 30], [28, 32],
  ];

  // Hand connections
  static const List<List<int>> handConnections = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle
    [0, 9], [9, 10], [10, 11], [11, 12],
    // Ring
    [0, 13], [13, 14], [14, 15], [15, 16],
    // Pinky
    [0, 17], [17, 18], [18, 19], [19, 20],
    // Palm
    [5, 9], [9, 13], [13, 17],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final posePaint = Paint()
      ..color = Colors.cyan
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    final posePointPaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 6
      ..strokeCap = StrokeCap.round;

    final leftHandPaint = Paint()
      ..color = Colors.orange
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round;

    final rightHandPaint = Paint()
      ..color = Colors.lightGreen
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round;

    final handPointPaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;

    // Draw pose
    if (frame.pose.isNotEmpty) {
      _drawConnections(canvas, size, frame.pose, poseConnections, posePaint);
      _drawPoints(canvas, size, frame.pose, posePointPaint);
    }

    // Draw left hand (connected to wrist at pose index 15)
    if (frame.leftHand.isNotEmpty) {
      _drawConnections(canvas, size, frame.leftHand, handConnections, leftHandPaint);
      _drawPoints(canvas, size, frame.leftHand, handPointPaint, pointSize: 3);
    }

    // Draw right hand (connected to wrist at pose index 16)
    if (frame.rightHand.isNotEmpty) {
      _drawConnections(canvas, size, frame.rightHand, handConnections, rightHandPaint);
      _drawPoints(canvas, size, frame.rightHand, handPointPaint, pointSize: 3);
    }
  }

  void _drawConnections(
    Canvas canvas,
    Size size,
    List<List<double>> landmarks,
    List<List<int>> connections,
    Paint paint,
  ) {
    for (final connection in connections) {
      if (connection[0] < landmarks.length && connection[1] < landmarks.length) {
        final p1 = landmarks[connection[0]];
        final p2 = landmarks[connection[1]];
        
        // Skip if any coordinate is invalid
        if (p1.length < 2 || p2.length < 2) continue;
        
        // Scale to canvas size (landmarks are normalized 0-1)
        final x1 = p1[0] * size.width;
        final y1 = p1[1] * size.height;
        final x2 = p2[0] * size.width;
        final y2 = p2[1] * size.height;
        
        canvas.drawLine(Offset(x1, y1), Offset(x2, y2), paint);
      }
    }
  }

  void _drawPoints(
    Canvas canvas,
    Size size,
    List<List<double>> landmarks,
    Paint paint, {
    double pointSize = 4,
  }) {
    for (final point in landmarks) {
      if (point.length < 2) continue;
      
      final x = point[0] * size.width;
      final y = point[1] * size.height;
      
      canvas.drawCircle(Offset(x, y), pointSize, paint);
    }
  }

  @override
  bool shouldRepaint(covariant LandmarkPainter oldDelegate) {
    return oldDelegate.frame != frame;
  }
}
