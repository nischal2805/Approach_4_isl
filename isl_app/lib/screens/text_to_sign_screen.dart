import 'package:flutter/material.dart';
import '../services/text_to_sign_service.dart';
import '../services/sign_animator.dart';
import '../widgets/skeleton_painter.dart';

/// Screen for Text-to-Sign translation
class TextToSignScreen extends StatefulWidget {
  const TextToSignScreen({super.key});

  @override
  State<TextToSignScreen> createState() => _TextToSignScreenState();
}

class _TextToSignScreenState extends State<TextToSignScreen> {
  final TextEditingController _textController = TextEditingController();
  final TextToSignService _textToSignService = TextToSignService();
  late SignAnimator _animator;
  
  bool _isInitialized = false;
  String _statusMessage = 'Initializing...';

  @override
  void initState() {
    super.initState();
    _animator = SignAnimator(_textToSignService);
    _animator.addListener(_onAnimatorUpdate);
    _initialize();
  }

  Future<void> _initialize() async {
    await _textToSignService.initialize();
    setState(() {
      _isInitialized = true;
      _statusMessage = '${_textToSignService.wordCount} signs loaded';
    });
  }

  void _onAnimatorUpdate() {
    if (mounted) setState(() {});
  }

  void _onTranslate() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;
    
    setState(() => _statusMessage = 'Loading signs...');
    
    await _animator.loadText(text);
    _animator.play();
    
    setState(() => _statusMessage = 'Playing...');
  }

  @override
  void dispose() {
    _animator.removeListener(_onAnimatorUpdate);
    _animator.dispose();
    _textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Text to Sign'),
        actions: [
          // Speed control
          PopupMenuButton<double>(
            icon: const Icon(Icons.speed),
            tooltip: 'Playback Speed',
            onSelected: _animator.setSpeed,
            itemBuilder: (context) => [
              const PopupMenuItem(value: 0.5, child: Text('0.5x')),
              const PopupMenuItem(value: 1.0, child: Text('1x')),
              const PopupMenuItem(value: 1.5, child: Text('1.5x')),
              const PopupMenuItem(value: 2.0, child: Text('2x')),
            ],
          ),
        ],
      ),
      body: Column(
        children: [
          // Avatar Display
          Expanded(
            flex: 3,
            child: Container(
              margin: const EdgeInsets.all(16),
              child: Stack(
                children: [
                  // Skeleton Avatar
                  SkeletonAvatar(
                    frame: _animator.currentFrame,
                    signType: _animator.currentSignType,
                  ),
                  
                  // Current Sign Label
                  if (_animator.currentLabel.isNotEmpty)
                    Positioned(
                      bottom: 16,
                      left: 0,
                      right: 0,
                      child: Center(
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 24,
                            vertical: 12,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(24),
                          ),
                          child: Text(
                            _animator.currentLabel.toUpperCase(),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                    ),
                  
                  // Progress indicator
                  if (_animator.totalSigns > 0)
                    Positioned(
                      top: 16,
                      right: 16,
                      child: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.black54,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          '${_animator.currentSignIndex + 1}/${_animator.totalSigns}',
                          style: const TextStyle(color: Colors.white),
                        ),
                      ),
                    ),
                  
                  // Loading indicator
                  if (_animator.isLoading)
                    const Center(
                      child: CircularProgressIndicator(color: Colors.white),
                    ),
                ],
              ),
            ),
          ),
          
          // Progress Bar
          if (_animator.totalSigns > 0)
            LinearProgressIndicator(
              value: _animator.progress,
              backgroundColor: Colors.grey.shade300,
              valueColor: AlwaysStoppedAnimation<Color>(
                Theme.of(context).primaryColor,
              ),
            ),
          
          // Controls
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.grey.shade100,
            child: Column(
              children: [
                // Status
                Text(
                  _statusMessage,
                  style: TextStyle(
                    color: Colors.grey.shade600,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 12),
                
                // Playback controls
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    IconButton(
                      icon: const Icon(Icons.stop),
                      onPressed: _animator.stop,
                      iconSize: 32,
                    ),
                    const SizedBox(width: 16),
                    IconButton(
                      icon: Icon(
                        _animator.isPlaying ? Icons.pause : Icons.play_arrow,
                      ),
                      onPressed: _animator.isPlaying
                          ? _animator.pause
                          : _animator.play,
                      iconSize: 48,
                      color: Theme.of(context).primaryColor,
                    ),
                    const SizedBox(width: 16),
                    Text(
                      '${_animator.playbackSpeed}x',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                
                // Text Input
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _textController,
                        decoration: InputDecoration(
                          hintText: 'Enter text to translate...',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          filled: true,
                          fillColor: Colors.white,
                        ),
                        onSubmitted: (_) => _onTranslate(),
                        textInputAction: TextInputAction.go,
                      ),
                    ),
                    const SizedBox(width: 12),
                    ElevatedButton(
                      onPressed: _isInitialized ? _onTranslate : null,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.all(16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: const Icon(Icons.translate),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
