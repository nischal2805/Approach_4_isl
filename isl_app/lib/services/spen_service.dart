import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';

/// S Pen Air Actions Service for Samsung Galaxy Tab S9
/// 
/// Receives S Pen gestures and button events from native Android
class SPenService {
  static const MethodChannel _channel = MethodChannel('com.example.isl_translator/spen');
  static const EventChannel _eventChannel = EventChannel('com.example.isl_translator/spen_events');
  
  static SPenService? _instance;
  static SPenService get instance => _instance ??= SPenService._();
  
  SPenService._();
  
  Stream<SPenEvent>? _eventStream;
  
  /// Stream of S Pen events (gestures, button presses, demo outputs)
  Stream<SPenEvent> get events {
    _eventStream ??= _eventChannel
        .receiveBroadcastStream()
        .map((event) => SPenEvent.fromMap(Map<String, dynamic>.from(event)));
    return _eventStream!;
  }
  
  /// Trigger a demo output (cycles through preset outputs)
  Future<String?> triggerDemo() async {
    try {
      final result = await _channel.invokeMethod<String>('triggerDemo');
      return result;
    } catch (e) {
      if (kDebugMode) debugPrint('SPen: triggerDemo error - $e');
      return null;
    }
  }
  
  /// Move to next demo output
  Future<void> nextDemo() async {
    try {
      await _channel.invokeMethod('nextDemo');
    } catch (e) {
      if (kDebugMode) debugPrint('SPen: nextDemo error - $e');
    }
  }
  
  /// Move to previous demo output
  Future<void> prevDemo() async {
    try {
      await _channel.invokeMethod('prevDemo');
    } catch (e) {
      if (kDebugMode) debugPrint('SPen: prevDemo error - $e');
    }
  }
}

/// Types of S Pen events
enum SPenEventType {
  gesture,    // Air Action gesture
  demoOutput, // Demo output triggered
  unknown,
}

/// S Pen gesture types
enum SPenGesture {
  buttonClick,
  buttonDouble,
  buttonLong,
  swipeLeft,
  swipeRight,
  swipeUp,
  swipeDown,
  circleCw,
  circleCcw,
  unknown,
}

/// S Pen event data
class SPenEvent {
  final SPenEventType type;
  final SPenGesture? gesture;
  final String? text;
  final int timestamp;
  
  SPenEvent({
    required this.type,
    this.gesture,
    this.text,
    required this.timestamp,
  });
  
  factory SPenEvent.fromMap(Map<String, dynamic> map) {
    final typeStr = map['type'] as String? ?? 'unknown';
    final type = typeStr == 'gesture' 
        ? SPenEventType.gesture 
        : typeStr == 'demo_output' 
            ? SPenEventType.demoOutput 
            : SPenEventType.unknown;
    
    SPenGesture? gesture;
    if (type == SPenEventType.gesture) {
      final gestureStr = map['gesture'] as String? ?? '';
      gesture = _parseGesture(gestureStr);
    }
    
    return SPenEvent(
      type: type,
      gesture: gesture,
      text: map['text'] as String?,
      timestamp: map['timestamp'] as int? ?? 0,
    );
  }
  
  static SPenGesture _parseGesture(String str) {
    switch (str) {
      case 'button_click': return SPenGesture.buttonClick;
      case 'button_double': return SPenGesture.buttonDouble;
      case 'button_long': return SPenGesture.buttonLong;
      case 'swipe_left': return SPenGesture.swipeLeft;
      case 'swipe_right': return SPenGesture.swipeRight;
      case 'swipe_up': return SPenGesture.swipeUp;
      case 'swipe_down': return SPenGesture.swipeDown;
      case 'circle_cw': return SPenGesture.circleCw;
      case 'circle_ccw': return SPenGesture.circleCcw;
      default: return SPenGesture.unknown;
    }
  }
  
  @override
  String toString() => 'SPenEvent(type: $type, gesture: $gesture, text: $text)';
}
