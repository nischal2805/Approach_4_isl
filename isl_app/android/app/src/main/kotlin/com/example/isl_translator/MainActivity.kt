package com.example.isl_translator

import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * Main Activity with Platform Channel for MediaPipe Holistic landmarks
 */
class MainActivity : FlutterActivity() {
    
    companion object {
        private const val CHANNEL = "com.example.isl_translator/landmarks"
        private const val TAG = "MainActivity"
    }
    
    private var landmarkExtractor: HolisticLandmarkExtractor? = null
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
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
    
    override fun onDestroy() {
        landmarkExtractor?.close()
        super.onDestroy()
    }
}
