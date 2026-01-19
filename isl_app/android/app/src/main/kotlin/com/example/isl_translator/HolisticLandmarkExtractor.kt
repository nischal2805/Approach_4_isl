package com.example.isl_translator

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

/**
 * Native MediaPipe Holistic Landmark Extractor
 * Extracts 75 keypoints: 33 pose + 21 left hand + 21 right hand
 * Runs efficiently on GPU via MediaPipe Tasks API
 */
class HolisticLandmarkExtractor(private val context: Context) {
    
    companion object {
        private const val TAG = "HolisticExtractor"
        private const val POSE_MODEL = "pose_landmarker_lite.task"
        private const val HAND_MODEL = "hand_landmarker.task"
        
        // Output dimensions
        const val POSE_LANDMARKS = 33
        const val HAND_LANDMARKS = 21
        const val TOTAL_LANDMARKS = 75  // 33 + 21 + 21
        const val FEATURES_PER_FRAME = 225  // 75 * 3 (x, y, z)
    }
    
    private var poseLandmarker: PoseLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private var isInitialized = false
    
    /**
     * Initialize MediaPipe models
     * Call this once before processing frames
     */
    fun initialize(): Boolean {
        return try {
            // Initialize Pose Landmarker
            val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(POSE_MODEL)
                        .setDelegate(Delegate.GPU)  // Use GPU for Snapdragon 8 Gen 2
                        .build()
                )
                .setRunningMode(RunningMode.IMAGE)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()
            
            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
            Log.d(TAG, "Pose Landmarker initialized with GPU")
            
            // Initialize Hand Landmarker
            val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(HAND_MODEL)
                        .setDelegate(Delegate.GPU)
                        .build()
                )
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(2)  // Detect both hands
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()
            
            handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
            Log.d(TAG, "Hand Landmarker initialized with GPU")
            
            isInitialized = true
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe: ${e.message}")
            e.printStackTrace()
            
            // Fallback to CPU if GPU fails
            try {
                initializeWithCpu()
            } catch (e2: Exception) {
                Log.e(TAG, "CPU fallback also failed: ${e2.message}")
                false
            }
        }
    }
    
    private fun initializeWithCpu(): Boolean {
        val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(POSE_MODEL)
                    .setDelegate(Delegate.CPU)
                    .build()
            )
            .setRunningMode(RunningMode.IMAGE)
            .setNumPoses(1)
            .build()
        
        poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
        
        val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(HAND_MODEL)
                    .setDelegate(Delegate.CPU)
                    .build()
            )
            .setRunningMode(RunningMode.IMAGE)
            .setNumHands(2)
            .build()
        
        handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
        
        Log.d(TAG, "Initialized with CPU fallback")
        isInitialized = true
        return true
    }
    
    /**
     * Process a camera frame and extract landmarks
     * 
     * @param imageBytes YUV420 image bytes from Flutter camera
     * @param width Image width
     * @param height Image height
     * @param rotation Camera rotation (0, 90, 180, 270)
     * @return DoubleArray of 225 values (75 landmarks * 3 coords) or empty if failed
     */
    fun processFrame(imageBytes: ByteArray, width: Int, height: Int, rotation: Int): DoubleArray {
        if (!isInitialized) {
            Log.w(TAG, "Not initialized, returning empty")
            return DoubleArray(0)
        }
        
        return try {
            // Convert YUV to Bitmap
            val bitmap = yuvToBitmap(imageBytes, width, height)
            if (bitmap == null) {
                Log.w(TAG, "Failed to convert YUV to bitmap")
                return DoubleArray(0)
            }
            
            // Rotate bitmap if needed
            val rotatedBitmap = rotateBitmap(bitmap, rotation)
            
            // Create MediaPipe image
            val mpImage = BitmapImageBuilder(rotatedBitmap).build()
            
            // Output array: 225 floats (75 landmarks * 3)
            val landmarks = DoubleArray(FEATURES_PER_FRAME) { 0.0 }
            
            // 1. Process Pose (33 landmarks -> indices 0-32)
            val poseResult = poseLandmarker?.detect(mpImage)
            if (poseResult != null && poseResult.landmarks().isNotEmpty()) {
                val poseLandmarks = poseResult.landmarks()[0]
                for (i in 0 until minOf(poseLandmarks.size, POSE_LANDMARKS)) {
                    val lm = poseLandmarks[i]
                    val offset = i * 3
                    landmarks[offset] = lm.x().toDouble()
                    landmarks[offset + 1] = lm.y().toDouble()
                    landmarks[offset + 2] = lm.z().toDouble()
                }
            }
            
            // 2. Process Hands (21 landmarks each -> indices 33-53 left, 54-74 right)
            val handResult = handLandmarker?.detect(mpImage)
            if (handResult != null && handResult.landmarks().isNotEmpty()) {
                // Process up to 2 hands
                for (handIdx in 0 until minOf(handResult.landmarks().size, 2)) {
                    val handLandmarks = handResult.landmarks()[handIdx]
                    
                    // Determine if left or right hand based on handedness
                    val isLeftHand = if (handResult.handednesses().size > handIdx) {
                        handResult.handednesses()[handIdx][0].categoryName().lowercase() == "left"
                    } else {
                        handIdx == 0  // Default: first hand is left
                    }
                    
                    // Left hand: indices 33-53, Right hand: indices 54-74
                    val startIdx = if (isLeftHand) POSE_LANDMARKS else POSE_LANDMARKS + HAND_LANDMARKS
                    
                    for (i in 0 until minOf(handLandmarks.size, HAND_LANDMARKS)) {
                        val lm = handLandmarks[i]
                        val offset = (startIdx + i) * 3
                        landmarks[offset] = lm.x().toDouble()
                        landmarks[offset + 1] = lm.y().toDouble()
                        landmarks[offset + 2] = lm.z().toDouble()
                    }
                }
            }
            
            // Clean up
            if (rotatedBitmap != bitmap) {
                rotatedBitmap.recycle()
            }
            bitmap.recycle()
            
            landmarks
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame: ${e.message}")
            DoubleArray(0)
        }
    }
    
    /**
     * Convert YUV420/NV21 bytes to Bitmap
     */
    private fun yuvToBitmap(yuvBytes: ByteArray, width: Int, height: Int): Bitmap? {
        return try {
            val yuvImage = YuvImage(yuvBytes, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
            val jpegBytes = out.toByteArray()
            BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "YUV to Bitmap conversion failed: ${e.message}")
            null
        }
    }
    
    /**
     * Rotate bitmap based on camera orientation
     */
    private fun rotateBitmap(bitmap: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return bitmap
        
        val matrix = android.graphics.Matrix()
        matrix.postRotate(rotation.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    
    /**
     * Release resources
     */
    fun close() {
        poseLandmarker?.close()
        handLandmarker?.close()
        poseLandmarker = null
        handLandmarker = null
        isInitialized = false
        Log.d(TAG, "Resources released")
    }
}
