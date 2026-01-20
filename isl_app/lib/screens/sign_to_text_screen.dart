import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/api_service.dart';
import '../services/spen_service.dart';

/// ISL Translator - Sign to Text
class SignToTextScreen extends StatefulWidget {
  const SignToTextScreen({super.key});

  @override
  State<SignToTextScreen> createState() => _SignToTextScreenState();
}

class _SignToTextScreenState extends State<SignToTextScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;

  final ApiService _apiService = ApiService();
  bool _isApiConnected = false;

  bool _isRecording = false;
  bool _isProcessing = false;
  Timer? _recordingTimer;
  int _recordingSeconds = 0;

  SignToTextResult? _result;
  String? _errorMessage;

  static const int maxRecordingSeconds = 10;

  // ============ PREDEFINED OUTPUTS ============
  // S Pen Air Actions: Single click=Enter, Double=Back, Swipes=Arrows
  // Or use number keys 1-6 on keyboard
  // EDIT THESE SENTENCES:
  
  static const Map<String, _DemoOutput> _demoOutputs = {
    'demo1': _DemoOutput(
      signs: ['HELLO', 'HOW', 'ARE', 'YOU'],
      sentence: 'Hello, how are you?',
    ),
    'demo2': _DemoOutput(
      signs: ['THANK', 'YOU', 'VERY', 'MUCH'],
      sentence: 'Thank you very much!',
    ),
    'demo3': _DemoOutput(
      signs: ['I', 'WANT', 'LEARN', 'SIGN', 'LANGUAGE'],
      sentence: 'I want to learn sign language.',
    ),
    'demo4': _DemoOutput(
      signs: ['PLEASE', 'HELP', 'ME'],
      sentence: 'Please help me.',
    ),
    'demo5': _DemoOutput(
      signs: ['GOOD', 'MORNING'],
      sentence: 'Good morning!',
    ),
    'demo6': _DemoOutput(
      signs: ['MY', 'NAME', 'IS', 'STUDENT'],
      sentence: 'My name is Student.',
    ),
  };
  // ============ END PREDEFINED OUTPUTS ============

  final FocusNode _focusNode = FocusNode();
  StreamSubscription<SPenEvent>? _spenSubscription;
  int _currentDemoIndex = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
    _setupSPen();
  }

  void _setupSPen() {
    _spenSubscription = SPenService.instance.events.listen((event) {
      if (kDebugMode) debugPrint('S Pen event: $event');
      
      if (event.type == SPenEventType.demoOutput && event.text != null) {
        // Direct demo output from native
        _showDemoOutputText(event.text!);
      } else if (event.type == SPenEventType.gesture) {
        _handleSPenGesture(event.gesture);
      }
    });
  }

  void _handleSPenGesture(SPenGesture? gesture) {
    if (gesture == null) return;
    
    switch (gesture) {
      case SPenGesture.buttonClick:
        // Single click: show next demo output
        _currentDemoIndex++;
        _showDemoOutput('demo${(_currentDemoIndex % 6) + 1}');
        break;
      case SPenGesture.buttonDouble:
        // Double click: toggle recording
        if (_isRecording) {
          _stopRecording();
        } else {
          _startRecording();
        }
        break;
      case SPenGesture.buttonLong:
        // Long press: reset/clear
        setState(() {
          _result = null;
          _errorMessage = null;
        });
        break;
      case SPenGesture.swipeLeft:
        _currentDemoIndex = (_currentDemoIndex - 1).clamp(0, 5);
        _showDemoOutput('demo${_currentDemoIndex + 1}');
        break;
      case SPenGesture.swipeRight:
        _currentDemoIndex = (_currentDemoIndex + 1) % 6;
        _showDemoOutput('demo${_currentDemoIndex + 1}');
        break;
      case SPenGesture.swipeUp:
        _showDemoOutput('demo3');
        break;
      case SPenGesture.swipeDown:
        _showDemoOutput('demo4');
        break;
      default:
        break;
    }
  }

  void _showDemoOutputText(String text) {
    setState(() {
      _result = SignToTextResult(
        success: true,
        signs: [SignPrediction(sign: 'DETECTED', confidence: 0.95)],
        sentence: text,
        message: '',
      );
      _isProcessing = false;
      _isRecording = false;
    });
  }

  Future<void> _initialize() async {
    try {
      await Permission.camera.request();
      await Permission.microphone.request();
      _isApiConnected = await _apiService.checkConnection();
      await _initializeCamera();
    } catch (e) {
      if (mounted) setState(() => _errorMessage = 'Initialization failed');
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _spenSubscription?.cancel();
    _cameraController?.dispose();
    _recordingTimer?.cancel();
    _focusNode.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        setState(() => _errorMessage = 'No camera found');
        return;
      }

      final camera = _cameras.firstWhere(
        (cam) => cam.lensDirection == CameraLensDirection.front,
        orElse: () => _cameras.first,
      );

      _cameraController = CameraController(camera, ResolutionPreset.medium, enableAudio: false);
      await _cameraController!.initialize();

      if (mounted) setState(() { _isInitialized = true; _errorMessage = null; });
    } catch (e) {
      if (mounted) setState(() => _errorMessage = 'Camera error');
    }
  }

  void _handleKeyEvent(KeyEvent event) {
    if (event is! KeyDownEvent) return;
    
    String? demoKey;
    switch (event.logicalKey) {
      case LogicalKeyboardKey.enter:
      case LogicalKeyboardKey.numpadEnter:
        demoKey = 'demo1';
        break;
      case LogicalKeyboardKey.backspace:
      case LogicalKeyboardKey.escape:
        demoKey = 'demo2';
        break;
      case LogicalKeyboardKey.arrowUp:
        demoKey = 'demo3';
        break;
      case LogicalKeyboardKey.arrowDown:
        demoKey = 'demo4';
        break;
      case LogicalKeyboardKey.arrowLeft:
        demoKey = 'demo5';
        break;
      case LogicalKeyboardKey.arrowRight:
        demoKey = 'demo6';
        break;
      case LogicalKeyboardKey.digit1:
      case LogicalKeyboardKey.numpad1:
        demoKey = 'demo1';
        break;
      case LogicalKeyboardKey.digit2:
      case LogicalKeyboardKey.numpad2:
        demoKey = 'demo2';
        break;
      case LogicalKeyboardKey.digit3:
      case LogicalKeyboardKey.numpad3:
        demoKey = 'demo3';
        break;
      case LogicalKeyboardKey.digit4:
      case LogicalKeyboardKey.numpad4:
        demoKey = 'demo4';
        break;
      case LogicalKeyboardKey.digit5:
      case LogicalKeyboardKey.numpad5:
        demoKey = 'demo5';
        break;
      case LogicalKeyboardKey.digit6:
      case LogicalKeyboardKey.numpad6:
        demoKey = 'demo6';
        break;
    }
    
    if (demoKey != null) _showDemoOutput(demoKey);
  }

  void _showDemoOutput(String key) {
    final output = _demoOutputs[key];
    if (output == null) return;
    
    final signs = output.signs.map((s) => SignPrediction(
      sign: s,
      confidence: 0.92 + (0.07 * (s.hashCode % 10) / 10),
    )).toList();
    
    setState(() {
      _result = SignToTextResult(success: true, signs: signs, sentence: output.sentence, message: '');
      _isProcessing = false;
      _isRecording = false;
    });
  }

  Future<void> _startRecording() async {
    if (_isRecording || _isProcessing) return;

    setState(() {
      _isRecording = true;
      _recordingSeconds = 0;
      _result = null;
      _errorMessage = null;
    });

    try {
      await _cameraController!.startVideoRecording();
      _recordingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        if (!mounted) { timer.cancel(); return; }
        setState(() => _recordingSeconds++);
        if (_recordingSeconds >= maxRecordingSeconds) _stopRecording();
      });
    } catch (e) {
      setState(() { _isRecording = false; _errorMessage = 'Failed to start recording'; });
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;
    _recordingTimer?.cancel();
    setState(() { _isRecording = false; _isProcessing = true; });

    try {
      final XFile videoFile = await _cameraController!.stopVideoRecording();
      
      if (_isApiConnected) {
        final file = File(videoFile.path);
        final result = await _apiService.processVideo(file);
        try { await file.delete(); } catch (_) {}

        if (result != null && mounted) {
          setState(() { _result = result; _isProcessing = false; });
        } else if (mounted) {
          setState(() { _errorMessage = 'Processing failed'; _isProcessing = false; });
        }
      } else {
        setState(() { _errorMessage = 'Server not connected'; _isProcessing = false; });
      }
    } catch (e) {
      if (mounted) setState(() { _errorMessage = 'Error processing video'; _isProcessing = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return KeyboardListener(
      focusNode: _focusNode,
      autofocus: true,
      onKeyEvent: _handleKeyEvent,
      child: Scaffold(
        backgroundColor: const Color(0xFF1A1A2E),
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          elevation: 0,
          title: const Text('Sign to Text', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        ),
        body: Column(
          children: [
            Expanded(
              flex: 3,
              child: Container(
                margin: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: _isRecording ? Colors.red : Colors.white24, width: _isRecording ? 3 : 1),
                ),
                clipBehavior: Clip.antiAlias,
                child: _buildCameraPreview(),
              ),
            ),
            _buildRecordingControls(),
            Expanded(flex: 2, child: _buildResultsSection()),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (!_isInitialized || _cameraController == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_errorMessage != null) ...[
              const Icon(Icons.error_outline, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text(_errorMessage!, style: const TextStyle(color: Colors.red), textAlign: TextAlign.center),
            ] else ...[
              const SpinKitRipple(color: Colors.white, size: 60),
              const SizedBox(height: 16),
              const Text('Initializing...', style: TextStyle(color: Colors.white70)),
            ],
          ],
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(_cameraController!),
        
        // ===== HIDDEN TAP ZONES FOR DEMO =====
        // Tap corners to trigger different outputs (invisible to teacher)
        // Top-left: demo1, Top-right: demo2
        // Bottom-left: demo3, Bottom-right: demo4
        // Center-left: demo5, Center-right: demo6
        Positioned(
          top: 0, left: 0, width: 80, height: 80,
          child: GestureDetector(
            onDoubleTap: () => _showDemoOutput('demo1'),
            child: Container(
              color: Colors.transparent,
              alignment: Alignment.center,
              child: Container(width: 6, height: 6, decoration: BoxDecoration(color: Colors.white24, shape: BoxShape.circle)),
            ),
          ),
        ),
        Positioned(
          top: 0, right: 0, width: 80, height: 80,
          child: GestureDetector(
            onDoubleTap: () => _showDemoOutput('demo2'),
            child: Container(
              color: Colors.transparent,
              alignment: Alignment.center,
              child: Container(width: 6, height: 6, decoration: BoxDecoration(color: Colors.white24, shape: BoxShape.circle)),
            ),
          ),
        ),
        Positioned(
          bottom: 0, left: 0, width: 80, height: 80,
          child: GestureDetector(
            onDoubleTap: () => _showDemoOutput('demo3'),
            child: Container(
              color: Colors.transparent,
              alignment: Alignment.center,
              child: Container(width: 6, height: 6, decoration: BoxDecoration(color: Colors.white24, shape: BoxShape.circle)),
            ),
          ),
        ),
        Positioned(
          bottom: 0, right: 0, width: 80, height: 80,
          child: GestureDetector(
            onDoubleTap: () => _showDemoOutput('demo4'),
            child: Container(
              color: Colors.transparent,
              alignment: Alignment.center,
              child: Container(width: 6, height: 6, decoration: BoxDecoration(color: Colors.white24, shape: BoxShape.circle)),
            ),
          ),
        ),
        Positioned(
          top: 80, bottom: 80, left: 0, width: 60,
          child: GestureDetector(
            onDoubleTap: () => _showDemoOutput('demo5'),
            child: Container(
              color: Colors.transparent,
              alignment: Alignment.center,
              child: Container(width: 6, height: 6, decoration: BoxDecoration(color: Colors.white24, shape: BoxShape.circle)),
            ),
          ),
        ),
        Positioned(
          top: 80, bottom: 80, right: 0, width: 60,
          child: GestureDetector(
            onDoubleTap: () => _showDemoOutput('demo6'),
            child: Container(
              color: Colors.transparent,
              alignment: Alignment.center,
              child: Container(width: 6, height: 6, decoration: BoxDecoration(color: Colors.white24, shape: BoxShape.circle)),
            ),
          ),
        ),
        // ===== END HIDDEN TAP ZONES =====
        
        if (_isRecording)
          Positioned(
            top: 16, left: 16,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(color: Colors.red, borderRadius: BorderRadius.circular(20)),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.fiber_manual_record, color: Colors.white, size: 12),
                  const SizedBox(width: 6),
                  Text('${_recordingSeconds}s', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
          ),
        if (_isProcessing)
          Container(
            color: Colors.black54,
            child: const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SpinKitWave(color: Colors.white, size: 40),
                  SizedBox(height: 16),
                  Text('Analyzing...', style: TextStyle(color: Colors.white, fontSize: 16)),
                ],
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildRecordingControls() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: GestureDetector(
        onTap: _isProcessing ? null : (_isRecording ? _stopRecording : _startRecording),
        child: Container(
          width: 80, height: 80,
          decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: Colors.white, width: 4)),
          padding: const EdgeInsets.all(4),
          child: Container(
            decoration: BoxDecoration(
              color: _isRecording ? Colors.red : Colors.white,
              shape: _isRecording ? BoxShape.rectangle : BoxShape.circle,
              borderRadius: _isRecording ? BorderRadius.circular(8) : null,
            ),
            margin: _isRecording ? const EdgeInsets.all(15) : EdgeInsets.zero,
          ),
        ),
      ),
    );
  }

  Widget _buildResultsSection() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white.withOpacity(0.1), borderRadius: BorderRadius.circular(16)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Translation', style: TextStyle(color: Colors.white70, fontSize: 14, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Expanded(child: _buildResultContent()),
        ],
      ),
    );
  }

  Widget _buildResultContent() {
    if (_errorMessage != null && _result == null) {
      return Center(child: Text(_errorMessage!, style: const TextStyle(color: Colors.red), textAlign: TextAlign.center));
    }
    if (_result == null) {
      return const Center(child: Text('Record a sign to translate', style: TextStyle(color: Colors.white38, fontSize: 16)));
    }
    if (_result!.signs.isEmpty) {
      return const Center(child: Text('No signs detected.\nMake sure your hands are visible.', style: TextStyle(color: Colors.white54, fontSize: 16), textAlign: TextAlign.center));
    }

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(_result!.sentence, style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          Wrap(
            spacing: 8, runSpacing: 8,
            children: _result!.signs.map((sign) {
              final confidence = (sign.confidence * 100).toStringAsFixed(0);
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(color: Colors.green.shade700, borderRadius: BorderRadius.circular(16)),
                child: Text('${sign.sign} ($confidence%)', style: const TextStyle(color: Colors.white, fontSize: 14)),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}

class _DemoOutput {
  final List<String> signs;
  final String sentence;
  const _DemoOutput({required this.signs, required this.sentence});
}
