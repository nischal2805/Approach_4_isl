import 'dart:async';
import 'package:flutter/material.dart';
import 'text_to_sign_service.dart';

/// Controller for playing sign animations
class SignAnimator extends ChangeNotifier {
  final TextToSignService _textToSignService;
  
  List<SignQueueItem> _queue = [];
  List<SignData> _loadedSigns = [];
  
  int _currentSignIndex = 0;
  int _currentFrameIndex = 0;
  Timer? _playbackTimer;
  
  bool _isPlaying = false;
  bool _isLoading = false;
  double _playbackSpeed = 1.0;
  
  SignFrame? _currentFrame;
  String _currentLabel = '';
  int _totalSigns = 0;

  SignAnimator(this._textToSignService);

  /// Load and prepare signs for playback
  Future<void> loadText(String text) async {
    stop();
    
    _isLoading = true;
    notifyListeners();
    
    try {
      // Process text to get sign queue
      _queue = _textToSignService.processText(text);
      _totalSigns = _queue.length;
      
      // Pre-load all signs
      _loadedSigns = [];
      for (var item in _queue) {
        final sign = await _textToSignService.loadSign(item);
        _loadedSigns.add(sign);
      }
      
      _currentSignIndex = 0;
      _currentFrameIndex = 0;
      
      if (_loadedSigns.isNotEmpty && _loadedSigns[0].frames.isNotEmpty) {
        _currentFrame = _loadedSigns[0].frames[0];
        _currentLabel = _loadedSigns[0].label;
      }
      
    } catch (e) {
      print('Error loading signs: $e');
    }
    
    _isLoading = false;
    notifyListeners();
  }

  /// Start playback
  void play() {
    if (_loadedSigns.isEmpty || _isPlaying) return;
    
    _isPlaying = true;
    _scheduleNextFrame();
    notifyListeners();
  }

  /// Pause playback
  void pause() {
    _playbackTimer?.cancel();
    _isPlaying = false;
    notifyListeners();
  }

  /// Stop and reset
  void stop() {
    _playbackTimer?.cancel();
    _isPlaying = false;
    _currentSignIndex = 0;
    _currentFrameIndex = 0;
    _currentFrame = null;
    _currentLabel = '';
    notifyListeners();
  }

  void _scheduleNextFrame() {
    if (!_isPlaying || _currentSignIndex >= _loadedSigns.length) {
      _isPlaying = false;
      notifyListeners();
      return;
    }
    
    SignData currentSign = _loadedSigns[_currentSignIndex];
    
    // Handle pause
    if (currentSign.isPause) {
      _currentLabel = '...';
      notifyListeners();
      
      _playbackTimer = Timer(
        Duration(milliseconds: (currentSign.pauseDuration * 1000 / _playbackSpeed).round()),
        () {
          _currentSignIndex++;
          _currentFrameIndex = 0;
          _scheduleNextFrame();
        },
      );
      return;
    }
    
    // Handle unknown sign
    if (currentSign.type == 'unknown') {
      _currentLabel = '[${currentSign.label}?]';
      _currentFrame = null;
      notifyListeners();
      
      _playbackTimer = Timer(
        Duration(milliseconds: (500 / _playbackSpeed).round()),
        () {
          _currentSignIndex++;
          _currentFrameIndex = 0;
          _scheduleNextFrame();
        },
      );
      return;
    }
    
    // Handle normal sign
    if (currentSign.frames.isEmpty) {
      _currentSignIndex++;
      _currentFrameIndex = 0;
      _scheduleNextFrame();
      return;
    }
    
    // Update current frame
    _currentFrame = currentSign.frames[_currentFrameIndex];
    _currentLabel = currentSign.label;
    notifyListeners();
    
    // Calculate frame duration
    double frameDuration = 1000 / (currentSign.fps * _playbackSpeed);
    
    _playbackTimer = Timer(
      Duration(milliseconds: frameDuration.round()),
      () {
        _currentFrameIndex++;
        
        if (_currentFrameIndex >= currentSign.frames.length) {
          // Move to next sign
          _currentSignIndex++;
          _currentFrameIndex = 0;
        }
        
        _scheduleNextFrame();
      },
    );
  }

  /// Set playback speed (0.5x, 1x, 2x)
  void setSpeed(double speed) {
    _playbackSpeed = speed;
    notifyListeners();
  }

  // Getters
  bool get isPlaying => _isPlaying;
  bool get isLoading => _isLoading;
  SignFrame? get currentFrame => _currentFrame;
  String get currentLabel => _currentLabel;
  double get playbackSpeed => _playbackSpeed;
  int get currentSignIndex => _currentSignIndex;
  int get totalSigns => _totalSigns;
  double get progress => _totalSigns > 0 ? _currentSignIndex / _totalSigns : 0;
  bool get isComplete => _currentSignIndex >= _loadedSigns.length && !_isPlaying;

  @override
  void dispose() {
    _playbackTimer?.cancel();
    super.dispose();
  }
}
