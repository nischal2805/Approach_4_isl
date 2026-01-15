import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/translation_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  bool _isRecording = false;
  bool _isTranslating = false;
  
  final TranslationService _translationService = TranslationService();
  final List<String> _translationHistory = [];
  String _currentTranslation = '';
  String? _errorMessage;
  
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
  }
  
  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
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
    // Request permissions
    final cameraStatus = await Permission.camera.request();
    final micStatus = await Permission.microphone.request();
    
    if (!cameraStatus.isGranted || !micStatus.isGranted) {
      setState(() {
        _errorMessage = 'Camera and microphone permissions required';
      });
      return;
    }
    
    try {
      _cameras = await availableCameras();
      
      if (_cameras.isEmpty) {
        setState(() {
          _errorMessage = 'No camera found';
        });
        return;
      }
      
      // Use front camera if available
      final camera = _cameras.firstWhere(
        (cam) => cam.lensDirection == CameraLensDirection.front,
        orElse: () => _cameras.first,
      );
      
      _cameraController = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      
      await _cameraController!.initialize();
      
      if (mounted) {
        setState(() {
          _isInitialized = true;
          _errorMessage = null;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to initialize camera: $e';
      });
    }
  }
  
  Future<void> _startRecording() async {
    if (_cameraController == null || _isRecording) return;
    
    try {
      await _cameraController!.startVideoRecording();
      setState(() {
        _isRecording = true;
        _currentTranslation = '';
        _errorMessage = null;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to start recording: $e';
      });
    }
  }
  
  Future<void> _stopRecordingAndTranslate() async {
    if (_cameraController == null || !_isRecording) return;
    
    try {
      final videoFile = await _cameraController!.stopVideoRecording();
      
      setState(() {
        _isRecording = false;
        _isTranslating = true;
      });
      
      // Send to API
      final result = await _translationService.translateVideo(
        File(videoFile.path),
      );
      
      setState(() {
        _isTranslating = false;
        
        if (result.success) {
          _currentTranslation = result.translation;
          if (result.translation.isNotEmpty) {
            _translationHistory.insert(0, result.translation);
            // Keep only last 10
            if (_translationHistory.length > 10) {
              _translationHistory.removeLast();
            }
          }
          _errorMessage = null;
        } else {
          _errorMessage = result.error;
        }
      });
      
      // Clean up video file
      try {
        await File(videoFile.path).delete();
      } catch (_) {}
      
    } catch (e) {
      setState(() {
        _isRecording = false;
        _isTranslating = false;
        _errorMessage = 'Translation failed: $e';
      });
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ISL Translator'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: _showSettings,
          ),
        ],
      ),
      body: Column(
        children: [
          // Camera preview
          Expanded(
            flex: 3,
            child: _buildCameraPreview(),
          ),
          
          // Translation result
          Expanded(
            flex: 2,
            child: _buildTranslationPanel(),
          ),
        ],
      ),
    );
  }
  
  Widget _buildCameraPreview() {
    if (!_isInitialized || _cameraController == null) {
      return Container(
        color: Colors.grey.shade100,
        child: Center(
          child: _errorMessage != null
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.error_outline, 
                         size: 48, 
                         color: Colors.grey.shade400),
                    const SizedBox(height: 16),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 32),
                      child: Text(
                        _errorMessage!,
                        textAlign: TextAlign.center,
                        style: TextStyle(color: Colors.grey.shade600),
                      ),
                    ),
                    const SizedBox(height: 16),
                    TextButton(
                      onPressed: _initializeCamera,
                      child: const Text('Retry'),
                    ),
                  ],
                )
              : const SpinKitDoubleBounce(
                  color: Color(0xFF1A1A2E),
                  size: 50,
                ),
        ),
      );
    }
    
    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera feed
        ClipRRect(
          child: CameraPreview(_cameraController!),
        ),
        
        // Recording indicator
        if (_isRecording)
          Positioned(
            top: 16,
            left: 16,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.red.shade600,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: const BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  const Text(
                    'Recording',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ),
        
        // Translating overlay
        if (_isTranslating)
          Container(
            color: Colors.black.withOpacity(0.5),
            child: const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SpinKitWave(
                    color: Colors.white,
                    size: 40,
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Translating...',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ),
        
        // Record button
        Positioned(
          bottom: 24,
          left: 0,
          right: 0,
          child: Center(
            child: GestureDetector(
              onTapDown: (_) => _startRecording(),
              onTapUp: (_) => _stopRecordingAndTranslate(),
              onTapCancel: () => _stopRecordingAndTranslate(),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 150),
                width: _isRecording ? 80 : 72,
                height: _isRecording ? 80 : 72,
                decoration: BoxDecoration(
                  color: _isRecording ? Colors.red : const Color(0xFF1A1A2E),
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: Colors.white,
                    width: 4,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.3),
                      blurRadius: 10,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Icon(
                  _isRecording ? Icons.stop : Icons.fiber_manual_record,
                  color: Colors.white,
                  size: 32,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
  
  Widget _buildTranslationPanel() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Handle
          Center(
            child: Container(
              margin: const EdgeInsets.only(top: 12),
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          
          // Current translation
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Translation',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: Colors.grey.shade600,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.grey.shade200),
                  ),
                  child: Text(
                    _currentTranslation.isEmpty 
                        ? 'Hold the button and sign to translate'
                        : _currentTranslation,
                    style: TextStyle(
                      fontSize: 18,
                      color: _currentTranslation.isEmpty 
                          ? Colors.grey.shade400 
                          : Colors.grey.shade800,
                      height: 1.4,
                    ),
                  ),
                ),
              ],
            ),
          ),
          
          // Error message
          if (_errorMessage != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _errorMessage!,
                  style: TextStyle(
                    color: Colors.red.shade700,
                    fontSize: 13,
                  ),
                ),
              ),
            ),
          
          // History
          if (_translationHistory.isNotEmpty)
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                itemCount: _translationHistory.length,
                itemBuilder: (context, index) {
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Text(
                      _translationHistory[index],
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade500,
                      ),
                    ),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }
  
  void _showSettings() {
    final controller = TextEditingController(
      text: TranslationService.baseUrl,
    );
    
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
          left: 24,
          right: 24,
          top: 24,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Settings',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              'API Endpoint',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                hintText: 'http://localhost:8000',
              ),
            ),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  _translationService.setBaseUrl(controller.text);
                  Navigator.pop(context);
                },
                child: const Text('Save'),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
