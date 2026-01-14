#!/usr/bin/env python3
"""
ISL Recognition System - Mobile Demo Template

This provides code templates for integrating the TFLite model with:
1. Python TFLite runtime
2. Android (Java/Kotlin)
3. MediaPipe Android SDK
"""

PYTHON_TFLITE_TEMPLATE = '''
# Python TFLite Inference Example
import numpy as np
import tensorflow as tf

class ISLRecognizer:
    def __init__(self, model_path: str, labels_path: str):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f]
    
    def predict(self, features: np.ndarray) -> tuple:
        """
        Predict sign from aggregated features.
        
        Args:
            features: numpy array of shape [675]
        
        Returns:
            (sign_name, confidence)
        """
        input_data = features.astype(np.float32).reshape(1, -1)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        pred_idx = np.argmax(output)
        confidence = output[pred_idx]
        
        return self.labels[pred_idx], float(confidence)

# Usage:
# recognizer = ISLRecognizer("models/model.tflite", "models/labels.txt")
# sign, conf = recognizer.predict(features)
'''

ANDROID_KOTLIN_TEMPLATE = '''
// Android Kotlin TFLite Inference
// Add to build.gradle: implementation 'org.tensorflow:tensorflow-lite:2.14.0'

import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ISLRecognizer(private val context: Context) {
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>
    
    init {
        // Load model from assets
        val model = loadModelFile("model.tflite")
        interpreter = Interpreter(model)
        
        // Load labels
        labels = context.assets.open("labels.txt")
            .bufferedReader().readLines()
    }
    
    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }
    
    fun predict(features: FloatArray): Pair<String, Float> {
        val inputArray = arrayOf(features)
        val outputArray = Array(1) { FloatArray(labels.size) }
        
        interpreter.run(inputArray, outputArray)
        
        val output = outputArray[0]
        val predIdx = output.indices.maxByOrNull { output[it] } ?: 0
        
        return Pair(labels[predIdx], output[predIdx])
    }
}

// Usage with MediaPipe Holistic:
/*
val holistic = Holistic(context, HolisticOptions.builder()
    .setStaticImageMode(false)
    .setMinDetectionConfidence(0.5f)
    .build())

val recognizer = ISLRecognizer(context)
val landmarkBuffer = mutableListOf<FloatArray>()

holistic.setResultListener { result ->
    val landmarks = extractLandmarks(result)  // Your function
    landmarkBuffer.add(landmarks)
    
    if (landmarkBuffer.size >= 30) {
        val aggregated = aggregateFeatures(landmarkBuffer)  // Mean, max, std
        val (sign, confidence) = recognizer.predict(aggregated)
        // Update UI
    }
}
*/
'''

MEDIAPIPE_ANDROID_TEMPLATE = '''
// MediaPipe Android Integration
// Add to build.gradle:
// implementation 'com.google.mediapipe:tasks-vision:0.10.9'

import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult

class MediaPipeLandmarkExtractor(context: Context) {
    private val landmarker: HolisticLandmarker
    
    init {
        val options = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder()
                .setModelAssetPath("holistic_landmarker.task")
                .build())
            .setMinPoseDetectionConfidence(0.5f)
            .setMinHandLandmarksConfidence(0.5f)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { result, _ -> processResult(result) }
            .build()
        
        landmarker = HolisticLandmarker.createFromOptions(context, options)
    }
    
    private fun processResult(result: HolisticLandmarkerResult) {
        val landmarks = FloatArray(225)  // 75 keypoints * 3 coords
        var idx = 0
        
        // Pose landmarks (33 * 3 = 99)
        result.poseLandmarks()?.let { pose ->
            for (lm in pose) {
                landmarks[idx++] = lm.x()
                landmarks[idx++] = lm.y()
                landmarks[idx++] = lm.z()
            }
        } ?: run { idx += 99 }
        
        // Left hand (21 * 3 = 63)
        result.leftHandLandmarks()?.let { hand ->
            for (lm in hand) {
                landmarks[idx++] = lm.x()
                landmarks[idx++] = lm.y()
                landmarks[idx++] = lm.z()
            }
        } ?: run { idx += 63 }
        
        // Right hand (21 * 3 = 63)
        result.rightHandLandmarks()?.let { hand ->
            for (lm in hand) {
                landmarks[idx++] = lm.x()
                landmarks[idx++] = lm.y()
                landmarks[idx++] = lm.z()
            }
        }
        
        // Add to buffer and run prediction...
    }
    
    fun aggregateFeatures(buffer: List<FloatArray>): FloatArray {
        val numFeatures = 225
        val result = FloatArray(675)  // mean + max + std
        
        // Mean
        for (i in 0 until numFeatures) {
            var sum = 0f
            for (frame in buffer) sum += frame[i]
            result[i] = sum / buffer.size
        }
        
        // Max
        for (i in 0 until numFeatures) {
            var maxVal = Float.MIN_VALUE
            for (frame in buffer) maxVal = maxOf(maxVal, frame[i])
            result[numFeatures + i] = maxVal
        }
        
        // Std
        for (i in 0 until numFeatures) {
            val mean = result[i]
            var sumSq = 0f
            for (frame in buffer) sumSq += (frame[i] - mean) * (frame[i] - mean)
            result[2 * numFeatures + i] = kotlin.math.sqrt(sumSq / buffer.size)
        }
        
        return result
    }
}
'''

def main():
    print("=" * 60)
    print("ISL Recognition - Mobile Integration Templates")
    print("=" * 60)
    
    print("\n1. PYTHON TFLITE TEMPLATE")
    print("-" * 40)
    print(PYTHON_TFLITE_TEMPLATE)
    
    print("\n2. ANDROID KOTLIN TEMPLATE")
    print("-" * 40)
    print(ANDROID_KOTLIN_TEMPLATE)
    
    print("\n3. MEDIAPIPE ANDROID TEMPLATE")
    print("-" * 40)
    print(MEDIAPIPE_ANDROID_TEMPLATE)
    
    print("\n" + "=" * 60)
    print("Files to include in Android app:")
    print("  - assets/model.tflite")
    print("  - assets/labels.txt")
    print("  - assets/holistic_landmarker.task (from MediaPipe)")
    print("=" * 60)


if __name__ == '__main__':
    main()
