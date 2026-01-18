import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/landmark_service.dart';
import '../services/sign_classifier.dart';
import '../services/grammar_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  bool _isProcessing = false;
  
  // Services
  final LandmarkService _landmarkService = LandmarkService();
  final SignClassifier _signClassifier = SignClassifier();
  final GrammarService _grammarService = GrammarService();
  
  // State
  String _currentWord = '';
  String _constructedSentence = '';
  bool _isGrammarCorrecting = false;
  double _confidence = 0.0;
  String? _errorMessage;
  
  // Buffering
  Timer? _pauseTimer;
  List<String> _wordBuffer = [];
  String _lastPredictedLabel = '';
  int _consecutiveFrames = 0;
  static const int CONSISTENCY_THRESHOLD = 5; // Frames to confirm a sign
  
  // Performance optimization
  int _frameCounter = 0;
  static const int PROCESS_EVERY_N_FRAMES = 3; // Skip frames for efficiency
  
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }
  
  Future<void> _initialize() async {
    await _requestPermissions();
    await _signClassifier.loadModel();
    await _grammarService.initialize(useLLM: true); // user settings can toggle this
    await _initializeCamera();
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
    _pauseTimer?.cancel();
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
      
      // Start Image Stream for Inference
      await _cameraController!.startImageStream(_processCameraImage);
      
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
  
  void _processCameraImage(CameraImage image) async {
    // Frame skipping for performance
    _frameCounter++;
    if (_frameCounter % PROCESS_EVERY_N_FRAMES != 0) return;
    
    if (_isProcessing) return;
    _isProcessing = true;
    
    try {
      // 1. Extract Landmarks
      final landmarks = await _landmarkService.extractLandmarks(image);
      
      if (landmarks.isEmpty) {
        // Handle no-sign / hands down
        _handleNoSign();
      } else {
        // 2. Classify Sign
        final result = await _signClassifier.processFrame(landmarks);
        
        if (result != null) {
          _handlePrediction(result);
        }
      }
    } catch (e) {
      debugPrint("Inference loop error: $e");
    } finally {
      _isProcessing = false;
    }
  }
  
  void _handlePrediction(Map<String, dynamic> result) {
    final String label = result['label'];
    final double conf = result['confidence'];
    
    // Reset pause timer on activity
    _pauseTimer?.cancel();
    _pauseTimer = Timer(const Duration(seconds: 2), _onPauseDetected);

    // Consistency Check
    if (label == _lastPredictedLabel) {
      _consecutiveFrames++;
    } else {
      _consecutiveFrames = 0;
      _lastPredictedLabel = label;
    }
    
    if (_consecutiveFrames > CONSISTENCY_THRESHOLD && conf > 0.6) {
      if (label != 'NO_SIGN' && label != _currentWord) {
        setState(() {
          _currentWord = label;
          _confidence = conf;
          
          // Add to buffer if different from last added
          if (_wordBuffer.isEmpty || _wordBuffer.last != label) {
            _wordBuffer.add(label);
            _updateConstructedSentence();
          }
        });
      } else if (label == 'NO_SIGN') {
         // Maybe clear current word display
         if (_currentWord.isNotEmpty) {
           setState(() => _currentWord = '');
         }
      }
    }
  }
  
  void _handleNoSign() {
     // If nothing detected for a while, maybe trigger pause logic?
     if (_pauseTimer == null || !_pauseTimer!.isActive) {
        _pauseTimer = Timer(const Duration(seconds: 2), _onPauseDetected);
     }
  }

  void _onPauseDetected() async {
    if (_wordBuffer.isNotEmpty) {
      final rawSentence = _wordBuffer.join(' ');
      _wordBuffer.clear(); // Clear buffer immediately
      
      if (mounted) {
        setState(() {
          _isGrammarCorrecting = true;
          _constructedSentence = "$rawSentence...";
          _currentWord = '';
        });
      }
      
      // Call Grammar Service
      final corrected = await _grammarService.correctGrammar(rawSentence);
      
      if (mounted) {
        setState(() {
           _constructedSentence = corrected;
           _isGrammarCorrecting = false;
        });
      }
    }
  }
  
  void _updateConstructedSentence() {
    // Show raw buffer as intermediate
    setState(() {
      _constructedSentence = _wordBuffer.join(' ') + '...';
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return const Scaffold(
        body: Center(child: SpinKitDoubleBounce(color: Color(0xFF1A1A2E))),
      );
    }
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live ISL Translator'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
               // Settings dialog
            },
          )
        ],
      ),
      body: Column(
        children: [
          // Camera Preview
          Expanded(
            flex: 3,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_cameraController!),
                
                // Prediction Overlay
                Positioned(
                  bottom: 20,
                  left: 20,
                  right: 20,
                  child: _currentWord.isNotEmpty 
                      ? Center(
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.7),
                              borderRadius: BorderRadius.circular(30),
                            ),
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(
                                  _currentWord,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 28,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                                Text(
                                  "${(_confidence * 100).toStringAsFixed(0)}%",
                                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                                )
                              ],
                            ),
                          ),
                        )
                      : const SizedBox.shrink(),
                ),
              ],
            ),
          ),
          
          // Translation Text Area
          Expanded(
            flex: 2,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              color: Colors.grey.shade50,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                   Text(
                    'Translation',
                    style: TextStyle(
                      color: Colors.grey.shade600,
                      fontWeight: FontWeight.w500,
                      letterSpacing: 1.0,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Expanded(
                    child: Container(
                       padding: const EdgeInsets.all(16),
                       decoration: BoxDecoration(
                         color: Colors.white,
                         borderRadius: BorderRadius.circular(16),
                         border: Border.all(color: Colors.grey.shade200),
                       ),
                       child: SingleChildScrollView(
                         child: Text(
                           _constructedSentence.isEmpty 
                               ? 'Start signing...' 
                               : _constructedSentence,
                           style: TextStyle(
                             fontSize: 22,
                             color: _constructedSentence.isEmpty 
                                 ? Colors.grey.shade400 
                                 : const Color(0xFF1A1A2E),
                             height: 1.5,
                           ),
                         ),
                       ),
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
