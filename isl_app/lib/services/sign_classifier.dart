import 'dart:convert';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../constants.dart';

/// Classifier for ISL signs using TFLite model
/// Processes sequences of frames (30 frames = 1 sign) with proper aggregation
class SignClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;
  int _numClasses = 0;
  
  /// Check if model is loaded
  bool get isLoaded => _isLoaded;
  
  // Scaler parameters (loaded from scaler.json)
  List<double>? _scalerMean;
  List<double>? _scalerStd;
  
  Future<void> loadModel() async {
    try {
      // Load TFLite model with GPU delegate for Snapdragon 8 Gen 2
      final options = InterpreterOptions();
      
      try {
        final gpuDelegate = GpuDelegate(options: GpuDelegateOptions());
        options.addDelegate(gpuDelegate);
        if (kDebugMode) debugPrint('SignClassifier: GPU delegate enabled');
      } catch (e) {
        if (kDebugMode) debugPrint('SignClassifier: GPU not available, using CPU: $e');
      }
      
      _interpreter = await Interpreter.fromAsset(AssetPaths.tfliteModel, options: options);
      
      // Get actual output shape from model
      final outputTensor = _interpreter!.getOutputTensor(0);
      _numClasses = outputTensor.shape[1]; // [1, num_classes]
      if (kDebugMode) debugPrint('SignClassifier: Model output classes: $_numClasses');
      
      // Load labels - trim each label to handle Windows line endings (\r\n)
      final labelData = await rootBundle.loadString(AssetPaths.labels);
      _labels = labelData.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();
      
      // Handle mismatch
      if (_labels.length != _numClasses) {
        if (kDebugMode) debugPrint('WARNING: Label count (${_labels.length}) != model classes ($_numClasses)');
        if (_labels.length > _numClasses) {
          _labels = _labels.sublist(0, _numClasses);
        }
      }
      
      // Load scaler parameters (REQUIRED for proper inference)
      try {
        final scalerData = await rootBundle.loadString(AssetPaths.scaler);
        final Map<String, dynamic> scaler = jsonDecode(scalerData);
        _scalerMean = List<double>.from(scaler['mean']);
        _scalerStd = List<double>.from(scaler['std']);
        
        if (_scalerMean!.length != ModelConstants.totalFeatures) {
          throw Exception('Scaler dimension mismatch: expected ${ModelConstants.totalFeatures}, got ${_scalerMean!.length}');
        }
        
        if (kDebugMode) debugPrint('SignClassifier: Scaler loaded (${_scalerMean!.length} features)');
      } catch (e) {
        if (kDebugMode) debugPrint('WARNING: Scaler loading failed: $e');
        // Continue without scaler - will use raw features
      }
      
      _isLoaded = true;
      if (kDebugMode) debugPrint('SignClassifier: Loaded with $_numClasses classes');
    } catch (e) {
      if (kDebugMode) debugPrint('SignClassifier: Error loading model: $e');
      rethrow;
    }
  }

  /// Classify a complete sequence of frames (for batch/recording mode)
  /// This is the proper way to use the model - matches training data
  /// 
  /// [frames] - List of landmark arrays, each with 225 values
  /// [mirrorForFrontCamera] - If true, flip x-coordinates to match training data
  /// Returns prediction with label, confidence, and index
  Future<Map<String, dynamic>?> classifySequence(List<List<double>> frames, {bool mirrorForFrontCamera = true}) async {
    if (!_isLoaded || frames.isEmpty) return null;
    
    // Validate frame dimensions
    for (int i = 0; i < frames.length; i++) {
      if (frames[i].length != ModelConstants.landmarksPerFrame) {
        if (kDebugMode) debugPrint('WARNING: Frame $i has ${frames[i].length} landmarks, expected ${ModelConstants.landmarksPerFrame}');
        return null;
      }
    }
    
    // Mirror coordinates for front camera (training used rear camera videos)
    var processedFrames = mirrorForFrontCamera 
        ? frames.map((f) => _mirrorLandmarks(f)).toList()
        : frames;
    
    // Aggregate features from all frames: mean, max, std
    var features = _aggregateFeatures(processedFrames);
    
    // Apply StandardScaler transformation
    features = _scaleFeatures(features);
    
    // Run inference
    return _predict(features);
  }
  
  /// Mirror landmarks for front camera to match training data format
  /// Training videos used rear camera (person's left hand on right side of frame)
  /// Front camera is mirrored (person's left hand on left side of frame)
  /// 
  /// This function:
  /// 1. Flips x-coordinates: x = 1.0 - x
  /// 2. Swaps left hand and right hand landmark positions
  List<double> _mirrorLandmarks(List<double> landmarks) {
    final result = List<double>.from(landmarks);
    
    // Constants
    const poseCount = 33;
    const handCount = 21;
    const coordsPerLandmark = 3;
    
    // 1. Flip all x-coordinates (every 3rd value starting at 0)
    for (int i = 0; i < result.length; i += coordsPerLandmark) {
      result[i] = 1.0 - result[i];  // x = 1.0 - x
    }
    
    // 2. Swap left hand (indices 99-161) with right hand (indices 162-224)
    // Left hand starts at: poseCount * 3 = 99
    // Right hand starts at: (poseCount + handCount) * 3 = 162
    const leftHandStart = poseCount * coordsPerLandmark;  // 99
    const rightHandStart = (poseCount + handCount) * coordsPerLandmark;  // 162
    const handDataSize = handCount * coordsPerLandmark;  // 63
    
    for (int i = 0; i < handDataSize; i++) {
      final temp = result[leftHandStart + i];
      result[leftHandStart + i] = result[rightHandStart + i];
      result[rightHandStart + i] = temp;
    }
    
    return result;
  }
  
  /// Apply StandardScaler transformation: (x - mean) / std
  List<double> _scaleFeatures(List<double> features) {
    if (_scalerMean == null || _scalerStd == null) {
      if (kDebugMode) debugPrint('WARNING: Scaler not loaded, using raw features!');
      return features;
    }
    
    final scaled = List.generate(features.length, (i) {
      double std = _scalerStd![i];
      if (std == 0) std = 1.0;
      double scaledVal = (features[i] - _scalerMean![i]) / std;
      // Handle NaN/Inf after scaling
      if (scaledVal.isNaN || scaledVal.isInfinite) return 0.0;
      return scaledVal;
    });
    
    // Debug: Verify scaling was applied correctly (should be ~mean=0, std=1)
    if (kDebugMode) {
      final scaledMin = scaled.reduce((a, b) => a < b ? a : b);
      final scaledMax = scaled.reduce((a, b) => a > b ? a : b);
      final sum = scaled.fold<double>(0, (a, b) => a + b);
      final scaledMean = sum / scaled.length;
      debugPrint('SignClassifier: SCALED features - range: [${scaledMin.toStringAsFixed(2)}, ${scaledMax.toStringAsFixed(2)}], mean: ${scaledMean.toStringAsFixed(4)}');
    }
    
    return scaled;
  }
  
  /// Aggregate frame landmarks into features: [mean, max, std]
  /// Input: [frames, 225] -> Output: [675]
  List<double> _aggregateFeatures(List<List<double>> frames) {
    final int numFrames = frames.length;
    const int dim = ModelConstants.landmarksPerFrame;
    
    final mean = List<double>.filled(dim, 0.0);
    final maxVals = List<double>.filled(dim, double.negativeInfinity);
    final stdDev = List<double>.filled(dim, 0.0);
    
    // 1. Calculate mean and max
    for (int i = 0; i < numFrames; i++) {
      for (int j = 0; j < dim; j++) {
        double val = frames[i][j];
        mean[j] += val;
        if (val > maxVals[j]) {
          maxVals[j] = val;
        }
      }
    }
    
    for (int j = 0; j < dim; j++) {
      mean[j] /= numFrames;
    }
    
    // 2. Calculate standard deviation
    for (int i = 0; i < numFrames; i++) {
      for (int j = 0; j < dim; j++) {
        double diff = frames[i][j] - mean[j];
        stdDev[j] += diff * diff;
      }
    }
    
    for (int j = 0; j < dim; j++) {
      stdDev[j] = sqrt(stdDev[j] / numFrames);
    }
    
    // Handle edge case: if max is still -infinity (no valid values), set to 0
    for (int j = 0; j < dim; j++) {
      if (maxVals[j] == double.negativeInfinity) {
        maxVals[j] = 0.0;
      }
    }
    
    // Concatenate: [mean (225), max (225), std (225)] = 675 features
    final aggregated = [...mean, ...maxVals, ...stdDev];
    
    // Handle NaN/Inf values (equivalent to np.nan_to_num)
    return aggregated.map((v) {
      if (v.isNaN || v.isInfinite) return 0.0;
      return v;
    }).toList();
  }
  
  /// Run TFLite inference with softmax
  Map<String, dynamic> _predict(List<double> features) {
    // Debug: Check feature statistics (only in debug mode)
    if (kDebugMode) {
      final nonZeroCount = features.where((f) => f != 0.0).length;
      final minVal = features.reduce((a, b) => a < b ? a : b);
      final maxVal = features.reduce((a, b) => a > b ? a : b);
      debugPrint('SignClassifier: Features - non-zero: $nonZeroCount/${ModelConstants.totalFeatures}, range: [${minVal.toStringAsFixed(2)}, ${maxVal.toStringAsFixed(2)}]');
    }
    
    // Input tensor: [1, 675]
    var input = [features];
    
    // Output tensor: [1, num_classes]
    var output = List.filled(1 * _numClasses, 0.0).reshape([1, _numClasses]);
    
    _interpreter!.run(input, output);
    
    // Get logits and apply softmax
    List<double> logits = List<double>.from(output[0]);
    
    // Debug: Check logit statistics (only in debug mode)
    if (kDebugMode) {
      final logitMin = logits.reduce((a, b) => a < b ? a : b);
      final logitMax = logits.reduce((a, b) => a > b ? a : b);
      debugPrint('SignClassifier: Logits range: [${logitMin.toStringAsFixed(2)}, ${logitMax.toStringAsFixed(2)}]');
    }
    
    List<double> probabilities = _softmax(logits);
    
    // Find best class
    double maxScore = -1.0;
    int maxIndex = -1;
    
    for (int i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxScore) {
        maxScore = probabilities[i];
        maxIndex = i;
      }
    }
    
    String label = (maxIndex >= 0 && maxIndex < _labels.length) 
        ? _labels[maxIndex] 
        : 'Unknown';
        
    return {
      'label': label,
      'confidence': maxScore,
      'index': maxIndex
    };
  }
  
  /// Softmax activation for proper probability distribution
  List<double> _softmax(List<double> logits) {
    // Find max for numerical stability
    double maxLogit = logits.reduce((a, b) => a > b ? a : b);
    
    // Compute exp(x - max) for stability
    List<double> expVals = logits.map((x) => exp(x - maxLogit)).toList();
    
    // Sum
    double sumExp = expVals.reduce((a, b) => a + b);
    
    // Normalize
    return expVals.map((x) => x / sumExp).toList();
  }
  
  void close() {
    _interpreter?.close();
  }
}
