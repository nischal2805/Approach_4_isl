/// App-wide constants for ISL Translator
library;

// ============== Model Constants ==============
class ModelConstants {
  // Landmark detection
  static const int poseLandmarks = 33;
  static const int handLandmarks = 21;
  static const int totalLandmarks = 75; // 33 + 21 + 21
  static const int landmarksPerFrame = 225; // 75 * 3 (x, y, z)
  static const int totalFeatures = 675; // 225 * 3 (mean, max, std)
  
  // Sign detection
  static const int signWindowFrames = 30; // Frames per sign detection window
  static const int slideStepFrames = 15; // Sliding window step
  static const double confidenceThreshold = 0.15; // Minimum confidence to accept
  
  // Recording
  static const int maxRecordingSeconds = 10;
  static const int targetFps = 30;
}

// ============== Landmark Constants ==============
class LandmarkConstants {
  /// Total expected landmarks per frame (75 keypoints * 3 coords)
  static const int expectedLandmarkCount = 225;
  
  /// Pose landmarks (body keypoints)
  static const int poseCount = 33;
  
  /// Hand landmarks (per hand)
  static const int handCount = 21;
  
  /// Total keypoints (pose + left hand + right hand)
  static const int totalKeypoints = 75;
}

// ============== Recording Constants ==============
class RecordingConstants {
  /// Maximum recording duration in seconds
  static const int maxDurationSeconds = 10;
  
  /// Number of frames that constitute one sign window
  static const int signWindowSize = 30;
  
  /// Sliding window step (frames)
  static const int slideStep = 15;
  
  /// Target frames per second
  static const int targetFps = 30;
}

// ============== Asset Paths ==============
class AssetPaths {
  // ML Models
  static const String tfliteModel = 'assets/model.tflite';
  static const String labels = 'assets/labels.txt';
  static const String scaler = 'assets/scaler.json';
  
  // LLM Model
  static const String llmModel = 'assets/SmolLM2-360M-Instruct-Q8_0.gguf';
  static const String llmModelSmall = 'assets/smollm2-135m-instruct-q8_0.gguf';
  
  // Sign reference images
  static const String signsFolder = 'assets/signs/';
  static const String wordsFolder = 'assets/signs/words/';
  static const String lettersFolder = 'assets/signs/letters/';
  static const String numbersFolder = 'assets/signs/numbers/';
  
  // 3D Avatar
  static const String avatarModel = 'assets/avatar/humanoid.glb';
}

// ============== LLM Settings ==============
class LLMSettings {
  static const int contextSize = 512;
  static const int batchSize = 256;
  static const int maxTokens = 50;
  static const int timeoutSeconds = 15;
  static const int maxHistoryItems = 5;
}

// ============== UI Constants ==============
class UIConstants {
  // Colors (for theme)
  static const int primaryColorHex = 0xFF6366F1;
  static const int accentColorHex = 0xFF8B5CF6;
  static const int backgroundDarkHex = 0xFF0F172A;
  static const int surfaceDarkHex = 0xFF1E293B;
  
  // Animations
  static const int defaultAnimationMs = 300;
  static const double cardBorderRadius = 16.0;
  static const double buttonBorderRadius = 12.0;
}

// ============== API Constants ==============
class APIConstants {
  // Translation API (for future online features)
  static const String baseUrl = 'https://api.isltranslator.com';
  static const int connectionTimeout = 30;
  static const int receiveTimeout = 60;
}
