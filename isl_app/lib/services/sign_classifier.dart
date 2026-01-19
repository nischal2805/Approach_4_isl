import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Classifier for ISL signs using TFLite model
/// Processes sequences of frames (30 frames = 1 sign) with proper aggregation
class SignClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;
  int _numClasses = 0;
  
  // Scaler parameters (loaded from scaler.json)
  List<double>? _scalerMean;
  List<double>? _scalerStd;
  
  // Feature dimensions
  static const int LANDMARKS_PER_FRAME = 225; // 75 landmarks * 3 coords
  static const int EXPECTED_FEATURES = 675;   // 225 * 3 (mean, max, std)
  
  Future<void> loadModel() async {
    try {
      // Load TFLite model with GPU delegate for Snapdragon 8 Gen 2
      final options = InterpreterOptions();
      
      try {
        final gpuDelegate = GpuDelegate(options: GpuDelegateOptions());
        options.addDelegate(gpuDelegate);
        print('SignClassifier: GPU delegate enabled');
      } catch (e) {
        print('SignClassifier: GPU not available, using CPU: $e');
      }
      
      _interpreter = await Interpreter.fromAsset('assets/model.tflite', options: options);
      
      // Get actual output shape from model
      final outputTensor = _interpreter!.getOutputTensor(0);
      _numClasses = outputTensor.shape[1]; // [1, num_classes]
      print('SignClassifier: Model output classes: $_numClasses');
      
      // Load labels
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').where((l) => l.isNotEmpty).toList();
      
      // Handle mismatch
      if (_labels.length != _numClasses) {
        print('WARNING: Label count (${_labels.length}) != model classes ($_numClasses)');
        if (_labels.length > _numClasses) {
          _labels = _labels.sublist(0, _numClasses);
        }
      }
      
      // Load scaler parameters (REQUIRED for proper inference)
      try {
        final scalerData = await rootBundle.loadString('assets/scaler.json');
        final Map<String, dynamic> scaler = jsonDecode(scalerData);
        _scalerMean = List<double>.from(scaler['mean']);
        _scalerStd = List<double>.from(scaler['std']);
        
        if (_scalerMean!.length != EXPECTED_FEATURES) {
          throw Exception('Scaler dimension mismatch: expected $EXPECTED_FEATURES, got ${_scalerMean!.length}');
        }
        
        print('SignClassifier: Scaler loaded (${_scalerMean!.length} features)');
      } catch (e) {
        print('WARNING: Scaler loading failed: $e');
        // Continue without scaler - will use raw features
      }
      
      _isLoaded = true;
      print('SignClassifier: Loaded with $_numClasses classes');
    } catch (e) {
      print('SignClassifier: Error loading model: $e');
      rethrow;
    }
  }

  /// Classify a complete sequence of frames (for batch/recording mode)
  /// This is the proper way to use the model - matches training data
  /// 
  /// [frames] - List of landmark arrays, each with 225 values
  /// Returns prediction with label, confidence, and index
  Future<Map<String, dynamic>?> classifySequence(List<List<double>> frames) async {
    if (!_isLoaded || frames.isEmpty) return null;
    
    // Validate frame dimensions
    for (int i = 0; i < frames.length; i++) {
      if (frames[i].length != LANDMARKS_PER_FRAME) {
        print('WARNING: Frame $i has ${frames[i].length} landmarks, expected $LANDMARKS_PER_FRAME');
        return null;
      }
    }
    
    // Aggregate features from all frames: mean, max, std
    var features = _aggregateFeatures(frames);
    
    // Apply StandardScaler transformation
    features = _scaleFeatures(features);
    
    // Run inference
    return _predict(features);
  }
  
  /// Apply StandardScaler transformation: (x - mean) / std
  List<double> _scaleFeatures(List<double> features) {
    if (_scalerMean == null || _scalerStd == null) {
      return features;
    }
    
    return List.generate(features.length, (i) {
      double std = _scalerStd![i];
      if (std == 0) std = 1.0;
      return (features[i] - _scalerMean![i]) / std;
    });
  }
  
  /// Aggregate frame landmarks into features: [mean, max, std]
  /// Input: [frames, 225] -> Output: [675]
  List<double> _aggregateFeatures(List<List<double>> frames) {
    final int numFrames = frames.length;
    const int dim = LANDMARKS_PER_FRAME;
    
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
    
    // Concatenate: [mean (225), max (225), std (225)] = 675 features
    return [...mean, ...maxVals, ...stdDev];
  }
  
  /// Run TFLite inference with softmax
  Map<String, dynamic> _predict(List<double> features) {
    // Input tensor: [1, 675]
    var input = [features];
    
    // Output tensor: [1, num_classes]
    var output = List.filled(1 * _numClasses, 0.0).reshape([1, _numClasses]);
    
    _interpreter!.run(input, output);
    
    // Get logits and apply softmax
    List<double> logits = List<double>.from(output[0]);
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
