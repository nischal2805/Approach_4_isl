import 'dart:async';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/landmark_service.dart';
import '../services/sign_classifier.dart';
import '../services/grammar_service.dart';

/// ISL Translator - Record → Process → Translate flow
/// Similar to Google Translate for sign language
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;

  // Services
  final LandmarkService _landmarkService = LandmarkService();
  final SignClassifier _signClassifier = SignClassifier();
  final GrammarService _grammarService = GrammarService();

  // Recording state
  bool _isRecording = false;
  bool _isProcessing = false;
  List<List<double>> _recordedFrames = [];
  Timer? _recordingTimer;
  int _recordingSeconds = 0;
  static const int MAX_RECORDING_SECONDS = 10;

  // Results
  String _glossOutput = '';
  String _translatedSentence = '';
  List<Map<String, dynamic>> _detectedSigns = [];
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  Future<void> _initialize() async {
    print('HomeScreen: Starting initialization...');
    await _requestPermissions();
    print('HomeScreen: Permissions granted');

    // Initialize native MediaPipe landmark extractor
    final landmarkSuccess = await _landmarkService.initialize();
    print('HomeScreen: LandmarkService initialized: $landmarkSuccess');
    if (!landmarkSuccess && mounted) {
      setState(() {
        _errorMessage = 'Failed to initialize MediaPipe. Check model files.';
      });
    }

    await _signClassifier.loadModel();
    print('HomeScreen: SignClassifier loaded');
    await _grammarService.initialize(useLLM: false);
    print('HomeScreen: GrammarService initialized');
    await _initializeCamera();
    print('HomeScreen: Camera initialized');
  }

  Future<void> _requestPermissions() async {
    await Permission.camera.request();
    await Permission.microphone.request();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _landmarkService.dispose();
    _signClassifier.close();
    _recordingTimer?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) return;

      final camera = _cameras.firstWhere(
        (cam) => cam.lensDirection == CameraLensDirection.front,
        orElse: () => _cameras.first,
      );

      _cameraController = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _cameraController!.initialize();

      // Pass sensor orientation to landmark service
      _landmarkService
          .setSensorOrientation(_cameraController!.description.sensorOrientation);

      if (mounted) {
        setState(() {
          _isInitialized = true;
          _errorMessage = null;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Camera error: $e';
      });
    }
  }

  /// Start recording sign language video
  void _startRecording() async {
    if (_isRecording || _isProcessing) return;

    setState(() {
      _isRecording = true;
      _recordedFrames.clear();
      _recordingSeconds = 0;
      _glossOutput = '';
      _translatedSentence = '';
      _detectedSigns.clear();
      _errorMessage = null;
    });

    // Start frame capture
    await _cameraController!.startImageStream(_captureFrame);

    // Start timer
    _recordingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        _recordingSeconds++;
      });

      if (_recordingSeconds >= MAX_RECORDING_SECONDS) {
        _stopRecording();
      }
    });
  }

  /// Capture frame landmarks during recording
  void _captureFrame(CameraImage image) async {
    if (!_isRecording) return;

    try {
      final landmarks = await _landmarkService.extractLandmarks(image);
      
      // Debug logging
      if (_recordedFrames.length % 10 == 0) {
        print('Frame ${_recordedFrames.length}: landmarks=${landmarks.length}, non-zero=${landmarks.where((v) => v != 0).length}');
      }
      
      if (landmarks.isNotEmpty && landmarks.length == 225) {
        _recordedFrames.add(landmarks);
      } else if (landmarks.isEmpty) {
        print('WARNING: Empty landmarks returned');
      }
    } catch (e) {
      print('Frame capture error: $e');
    }
  }

  /// Stop recording and process the video
  void _stopRecording() async {
    if (!_isRecording) return;

    _recordingTimer?.cancel();
    await _cameraController?.stopImageStream();

    setState(() {
      _isRecording = false;
      _isProcessing = true;
    });

    // Process recorded frames
    await _processRecordedVideo();

    setState(() {
      _isProcessing = false;
    });
  }

  /// Process recorded frames to detect signs
  Future<void> _processRecordedVideo() async {
    print('DEBUG: Processing video with ${_recordedFrames.length} frames');
    
    if (_recordedFrames.isEmpty) {
      setState(() {
        _errorMessage = 'No frames captured. Make sure your hands are visible.';
      });
      return;
    }

    print('Processing ${_recordedFrames.length} frames...');

    try {
      // Segment the recording into sign windows
      final signs = await _detectSignsInRecording();

      if (signs.isEmpty) {
        setState(() {
          _glossOutput = 'NO_SIGN';
          _translatedSentence = 'No signs detected. Please try again.';
        });
        return;
      }

      // Build gloss output
      final glossWords = signs.map((s) => s['label'] as String).toList();
      final gloss = glossWords.join(' ');

      setState(() {
        _detectedSigns = signs;
        _glossOutput = gloss;
      });

      // Grammar correction
      final translated = await _grammarService.correctGrammar(gloss);

      setState(() {
        _translatedSentence = translated;
      });

      print('Gloss: $gloss');
      print('Translation: $translated');
    } catch (e) {
      setState(() {
        _errorMessage = 'Processing error: $e';
      });
    }
  }

  /// Detect signs using sliding window over recorded frames
  Future<List<Map<String, dynamic>>> _detectSignsInRecording() async {
    final List<Map<String, dynamic>> detectedSigns = [];
    
    const int windowSize = 30; // 30 frames = 1 sign (matches training)
    const int stepSize = 15;   // 50% overlap for better detection
    
    print('DEBUG: Detecting signs in ${_recordedFrames.length} frames (window=$windowSize, step=$stepSize)');
    
    if (_recordedFrames.length < windowSize) {
      print('DEBUG: Not enough frames, processing ${_recordedFrames.length} frames directly');
      // Not enough frames, process what we have
      final result = await _signClassifier.classifySequence(_recordedFrames);
      print('DEBUG: Result: $result');
      if (result != null && result['label'] != 'NO_SIGN' && result['confidence'] > 0.2) {
        detectedSigns.add(result);
      }
      return detectedSigns;
    }

    // Sliding window detection
    String? lastSign;
    int windowCount = 0;
    for (int start = 0; start + windowSize <= _recordedFrames.length; start += stepSize) {
      final window = _recordedFrames.sublist(start, start + windowSize);
      windowCount++;
      
      final result = await _signClassifier.classifySequence(window);
      print('DEBUG: Window $windowCount (frames $start-${start+windowSize}): ${result?['label']} @ ${((result?['confidence'] ?? 0) * 100).toStringAsFixed(1)}%');
      
      if (result != null && 
          result['label'] != 'NO_SIGN' && 
          result['confidence'] > 0.2) {
        
        // Avoid consecutive duplicates
        if (result['label'] != lastSign) {
          detectedSigns.add(result);
          lastSign = result['label'];
          print('DEBUG: Added sign: ${result['label']}');
        }
      }
    }
    
    print('DEBUG: Total detected signs: ${detectedSigns.length}');

    return detectedSigns;
  }

  /// Clear results and prepare for new recording
  void _clearResults() {
    setState(() {
      _recordedFrames.clear();
      _glossOutput = '';
      _translatedSentence = '';
      _detectedSigns.clear();
      _errorMessage = null;
      _recordingSeconds = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return Scaffold(
        backgroundColor: const Color(0xFF1A1A2E),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const SpinKitDoubleBounce(color: Colors.white, size: 60),
              const SizedBox(height: 24),
              Text(
                'Initializing...',
                style: TextStyle(color: Colors.white.withOpacity(0.7), fontSize: 16),
              ),
              if (_errorMessage != null) ...[
                const SizedBox(height: 16),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 32),
                  child: Text(
                    _errorMessage!,
                    style: const TextStyle(color: Colors.redAccent, fontSize: 14),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('ISL Translator'),
        centerTitle: true,
        backgroundColor: const Color(0xFF1A1A2E),
        foregroundColor: Colors.white,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _clearResults,
            tooltip: 'Clear',
          ),
        ],
      ),
      body: Column(
        children: [
          // Camera Preview with Recording Overlay
          Expanded(
            flex: 3,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Camera
                ClipRRect(
                  child: CameraPreview(_cameraController!),
                ),

                // Recording indicator
                if (_isRecording)
                  Positioned(
                    top: 20,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                        decoration: BoxDecoration(
                          color: Colors.red.withOpacity(0.9),
                          borderRadius: BorderRadius.circular(30),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Container(
                              width: 12,
                              height: 12,
                              decoration: const BoxDecoration(
                                color: Colors.white,
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 10),
                            Text(
                              'Recording $_recordingSeconds/$MAX_RECORDING_SECONDS s',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),

                // Processing indicator
                if (_isProcessing)
                  Container(
                    color: Colors.black.withOpacity(0.7),
                    child: const Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SpinKitRipple(color: Colors.white, size: 80),
                          SizedBox(height: 20),
                          Text(
                            'Processing signs...',
                            style: TextStyle(color: Colors.white, fontSize: 18),
                          ),
                        ],
                      ),
                    ),
                  ),

                // Frame count indicator
                if (_isRecording)
                  Positioned(
                    bottom: 20,
                    left: 20,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.6),
                        borderRadius: BorderRadius.circular(15),
                      ),
                      child: Text(
                        '${_recordedFrames.length} frames',
                        style: const TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                    ),
                  ),
              ],
            ),
          ),

          // Record Button
          Container(
            padding: const EdgeInsets.symmetric(vertical: 20),
            color: const Color(0xFF1A1A2E),
            child: Center(
              child: GestureDetector(
                onTapDown: (_) => _startRecording(),
                onTapUp: (_) => _stopRecording(),
                onTapCancel: () => _stopRecording(),
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 4),
                    color: _isRecording ? Colors.red : Colors.transparent,
                  ),
                  child: Center(
                    child: Container(
                      width: 60,
                      height: 60,
                      decoration: BoxDecoration(
                        shape: _isRecording ? BoxShape.rectangle : BoxShape.circle,
                        borderRadius: _isRecording ? BorderRadius.circular(8) : null,
                        color: _isRecording ? Colors.red.shade700 : Colors.red,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),

          // Translation Results
          Expanded(
            flex: 2,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              color: Colors.grey.shade100,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Gloss output (ISL words)
                  if (_glossOutput.isNotEmpty) ...[
                    Text(
                      'Signs Detected:',
                      style: TextStyle(
                        color: Colors.grey.shade600,
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _detectedSigns.map((sign) {
                        return Chip(
                          label: Text(
                            '${sign['label']} (${(sign['confidence'] * 100).toStringAsFixed(0)}%)',
                            style: const TextStyle(fontSize: 12),
                          ),
                          backgroundColor: Colors.blue.shade100,
                        );
                      }).toList(),
                    ),
                    const SizedBox(height: 12),
                  ],

                  // Translation
                  Text(
                    'Translation:',
                    style: TextStyle(
                      color: Colors.grey.shade600,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Expanded(
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.05),
                            blurRadius: 10,
                            offset: const Offset(0, 2),
                          ),
                        ],
                      ),
                      child: SingleChildScrollView(
                        child: Text(
                          _translatedSentence.isEmpty
                              ? 'Hold the button and sign...'
                              : _translatedSentence,
                          style: TextStyle(
                            fontSize: 20,
                            color: _translatedSentence.isEmpty
                                ? Colors.grey.shade400
                                : const Color(0xFF1A1A2E),
                            height: 1.5,
                          ),
                        ),
                      ),
                    ),
                  ),

                  // Error message
                  if (_errorMessage != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(
                        _errorMessage!,
                        style: const TextStyle(color: Colors.redAccent, fontSize: 12),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
