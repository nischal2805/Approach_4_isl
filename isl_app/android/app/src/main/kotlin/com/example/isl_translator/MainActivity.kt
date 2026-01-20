package com.example.isl_translator

import android.util.Log
import android.view.KeyEvent
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.EventChannel

/**
 * Main Activity with Platform Channel for MediaPipe Holistic landmarks
 * and S Pen Air Actions support
 */
class MainActivity : FlutterActivity() {
    
    companion object {
        private const val CHANNEL = "com.example.isl_translator/landmarks"
        private const val SPEN_CHANNEL = "com.example.isl_translator/spen"
        private const val SPEN_EVENTS = "com.example.isl_translator/spen_events"
        private const val TAG = "MainActivity"
        
        // Demo outputs for S Pen gestures
        private val DEMO_OUTPUTS = listOf(
            "Hello, how are you?",
            "My name is [Your Name]",
            "Nice to meet you",
            "Thank you very much",
            "I am learning sign language",
            "Good morning"
        )
    }
    
    private var landmarkExtractor: HolisticLandmarkExtractor? = null
    private var spenHandler: SPenHandler? = null
    private var demoIndex = 0
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        // Initialize S Pen handler
        spenHandler = SPenHandler(this)
        
        // S Pen event channel (for receiving gestures in Flutter)
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, SPEN_EVENTS)
            .setStreamHandler(spenHandler)
        
        // S Pen method channel (for Flutter to trigger demo outputs)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, SPEN_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "triggerDemo" -> {
                    val text = DEMO_OUTPUTS[demoIndex % DEMO_OUTPUTS.size]
                    demoIndex++
                    spenHandler?.sendDemoOutput(text)
                    result.success(text)
                }
                "nextDemo" -> {
                    demoIndex++
                    result.success(demoIndex)
                }
                "prevDemo" -> {
                    demoIndex = if (demoIndex > 0) demoIndex - 1 else DEMO_OUTPUTS.size - 1
                    result.success(demoIndex)
                }
                "setDemoTexts" -> {
                    // Allow Flutter to set custom demo texts
                    result.success(true)
                }
                else -> result.notImplemented()
            }
        }
        
        // Landmarks channel (existing)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "initialize" -> {
                    try {
                        if (landmarkExtractor == null) {
                            landmarkExtractor = HolisticLandmarkExtractor(this)
                        }
                        val success = landmarkExtractor!!.initialize()
                        result.success(success)
                        Log.d(TAG, "Initialize called, success: $success")
                    } catch (e: Exception) {
                        Log.e(TAG, "Initialize error: ${e.message}")
                        result.error("INIT_ERROR", e.message, null)
                    }
                }
                
                "processFrame" -> {
                    try {
                        val imageBytes = call.argument<ByteArray>("bytes")
                        val width = call.argument<Int>("width") ?: 0
                        val height = call.argument<Int>("height") ?: 0
                        val rotation = call.argument<Int>("rotation") ?: 0
                        
                        if (imageBytes == null || width == 0 || height == 0) {
                            result.error("INVALID_ARGS", "Missing image data", null)
                            return@setMethodCallHandler
                        }
                        
                        if (landmarkExtractor == null) {
                            result.error("NOT_INIT", "Extractor not initialized", null)
                            return@setMethodCallHandler
                        }
                        
                        val landmarks = landmarkExtractor!!.processFrame(imageBytes, width, height, rotation)
                        
                        // Convert to List<Double> for Flutter
                        result.success(landmarks.toList())
                    } catch (e: Exception) {
                        Log.e(TAG, "ProcessFrame error: ${e.message}")
                        result.error("PROCESS_ERROR", e.message, null)
                    }
                }
                
                "dispose" -> {
                    try {
                        landmarkExtractor?.close()
                        landmarkExtractor = null
                        result.success(true)
                        Log.d(TAG, "Disposed")
                    } catch (e: Exception) {
                        result.error("DISPOSE_ERROR", e.message, null)
                    }
                }
                
                else -> result.notImplemented()
            }
        }
    }
    
    /**
     * Intercept key events for S Pen button detection
     */
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (event != null && spenHandler?.handleKeyEvent(keyCode, event) == true) {
            return true
        }
        return super.onKeyDown(keyCode, event)
    }
    
    override fun onKeyUp(keyCode: Int, event: KeyEvent?): Boolean {
        if (event != null && spenHandler?.handleKeyEvent(keyCode, event) == true) {
            return true
        }
        return super.onKeyUp(keyCode, event)
    }
    
    override fun onDestroy() {
        landmarkExtractor?.close()
        super.onDestroy()
    }
}
