import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import '../services/api_service.dart';

/// Text to Sign Screen - API-based
/// Converts English text to ISL sign animations using frame slideshow
class TextToSignApiScreen extends StatefulWidget {
  const TextToSignApiScreen({super.key});

  @override
  State<TextToSignApiScreen> createState() => _TextToSignApiScreenState();
}

class _TextToSignApiScreenState extends State<TextToSignApiScreen> {
  final TextEditingController _textController = TextEditingController();
  final ApiService _apiService = ApiService();

  bool _isLoading = false;
  bool _isApiConnected = false;
  String? _errorMessage;

  // Animation data
  AnimationResult? _animationData;
  List<AnimationFrame> _allFrames = [];
  List<String> _wordLabels = [];  // Maps frame index to word
  int _currentFrameIndex = 0;
  Timer? _animationTimer;
  bool _isPlaying = false;

  // Available words
  List<String> _availableWords = [];

  // Playback speed (ms per frame)
  int _frameDelay = 200;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    _isApiConnected = await _apiService.checkConnection();
    if (_isApiConnected) {
      _availableWords = await _apiService.getAvailableWords();
      if (kDebugMode) {
        debugPrint('TextToSign: Connected, ${_availableWords.length} words available');
      }
    }
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _animationTimer?.cancel();
    _textController.dispose();
    super.dispose();
  }

  Future<void> _translateText() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _stopAnimation();
      _allFrames.clear();
      _wordLabels.clear();
      _animationData = null;
    });

    try {
      final result = await _apiService.getAnimationData(text);
      
      if (result == null) {
        setState(() {
          _errorMessage = _apiService.lastError ?? 'Failed to get animation';
          _isLoading = false;
        });
        return;
      }

      // Flatten all frames with word labels
      final List<AnimationFrame> frames = [];
      final List<String> labels = [];
      
      for (final anim in result.animations) {
        if (anim.found) {
          for (final frame in anim.frames) {
            frames.add(frame);
            labels.add(anim.word);
          }
        }
      }

      setState(() {
        _animationData = result;
        _allFrames = frames;
        _wordLabels = labels;
        _currentFrameIndex = 0;
        _isLoading = false;
      });

      if (_allFrames.isNotEmpty) {
        _startAnimation();
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
          'Text to Sign',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Icon(
              _isApiConnected ? Icons.cloud_done : Icons.cloud_off,
              color: _isApiConnected ? Colors.green : Colors.red,
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
            const Icon(Icons.error_outline, color: Colors.red, size: 48),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              style: const TextStyle(color: Colors.red),
              textAlign: TextAlign.center,
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
                'Try: ${_availableWords.take(5).join(", ")}...',
                style: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 12),
              ),
            ],
          ],
        ),
      );
    }

    // Show current frame
    final currentFrame = _allFrames[_currentFrameIndex];
    final currentWord = _currentFrameIndex < _wordLabels.length 
        ? _wordLabels[_currentFrameIndex] 
        : '';

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white24),
      ),
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Frame image
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.memory(
              currentFrame.data,
              fit: BoxFit.contain,
              gaplessPlayback: true,  // Prevents flicker
            ),
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
                currentWord.toUpperCase(),
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
                style: const TextStyle(color: Colors.white70),
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
          // Progress bar
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: Colors.white,
              inactiveTrackColor: Colors.white24,
              thumbColor: Colors.white,
              overlayColor: Colors.white.withOpacity(0.2),
            ),
            child: Slider(
              value: _currentFrameIndex.toDouble(),
              min: 0,
              max: (_allFrames.length - 1).toDouble(),
              onChanged: (value) {
                _stopAnimation();
                setState(() {
                  _currentFrameIndex = value.toInt();
                });
              },
            ),
          ),

          // Control buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Speed control
              IconButton(
                icon: const Icon(Icons.speed, color: Colors.white70),
                onPressed: () {
                  setState(() {
                    // Cycle through speeds: 200 -> 150 -> 100 -> 300 -> 200
                    _frameDelay = switch (_frameDelay) {
                      200 => 150,
                      150 => 100,
                      100 => 300,
                      _ => 200,
                    };
                    if (_isPlaying) {
                      _stopAnimation();
                      _startAnimation();
                    }
                  });
                },
                tooltip: 'Speed: ${1000 ~/ _frameDelay} fps',
              ),
              
              const SizedBox(width: 16),

              // Previous
              IconButton(
                icon: const Icon(Icons.skip_previous, color: Colors.white, size: 32),
                onPressed: _previousFrame,
              ),

              // Play/Pause
              Container(
                width: 64,
                height: 64,
                decoration: const BoxDecoration(
                  color: Colors.white,
                  shape: BoxShape.circle,
                ),
                child: IconButton(
                  icon: Icon(
                    _isPlaying ? Icons.pause : Icons.play_arrow,
                    color: const Color(0xFF1A1A2E),
                    size: 32,
                  ),
                  onPressed: _togglePlayPause,
                ),
              ),

              // Next
              IconButton(
                icon: const Icon(Icons.skip_next, color: Colors.white, size: 32),
                onPressed: _nextFrame,
              ),
              
              const SizedBox(width: 16),

              // Restart
              IconButton(
                icon: const Icon(Icons.replay, color: Colors.white70),
                onPressed: () {
                  setState(() {
                    _currentFrameIndex = 0;
                  });
                  _startAnimation();
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildWordStatus() {
    if (_animationData == null) return const SizedBox.shrink();

    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          children: _animationData!.animations.map((anim) {
            return Container(
              margin: const EdgeInsets.only(right: 8),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: anim.found ? Colors.green.shade700 : Colors.red.shade700,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    anim.found ? Icons.check : Icons.close,
                    color: Colors.white,
                    size: 16,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    anim.word,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                  ),
                  if (anim.found) ...[
                    const SizedBox(width: 4),
                    Text(
                      '(${anim.frames.length})',
                      style: const TextStyle(color: Colors.white70, fontSize: 10),
                    ),
                  ],
                ],
              ),
            );
          }).toList(),
        ),
      ),
    );
  }
}
