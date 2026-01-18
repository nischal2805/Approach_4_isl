import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Classifier for ISL signs using TFLite model
class SignClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;
  
  // Scaler parameters (loaded from scaler.json)
  List<double>? _scalerMean;
  List<double>? _scalerStd;
  
  // Sliding window buffer
  final List<List<double>> _landmarkBuffer = [];
  static const int TARGET_FRAMES = 30; // 1 second buffer
  static const int LANDMARKS_PER_FRAME = 225;
  
  Future<void> loadModel() async {
    try {
      // Load TFLite model
      final options = InterpreterOptions();
      // Use XNNPACK or GPU delegate if supported
      // options.addDelegate(XNNPackDelegate()); 
      
      _interpreter = await Interpreter.fromAsset('assets/model.tflite', options: options);
      
      // Load labels
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').where((l) => l.isNotEmpty).toList();
      
      // Load scaler parameters (optional - if not found, skip scaling)
      try {
        final scalerData = await rootBundle.loadString('assets/scaler.json');
        final Map<String, dynamic> scaler = jsonDecode(scalerData);
        _scalerMean = List<double>.from(scaler['mean']);
        _scalerStd = List<double>.from(scaler['std']);
        print('Scaler loaded: ${_scalerMean!.length} features');
      } catch (e) {
        print('Scaler not found, using raw features: $e');
      }
      
      _isLoaded = true;
      print('SignClassifier loaded: ${_labels.length} classes');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  /// Process new frame landmarks and return prediction if ready
  Future<Map<String, dynamic>?> processFrame(List<double> landmarks) async {
    if (!_isLoaded) return null;
    
    // Add to buffer
    _landmarkBuffer.add(landmarks);
    
    // Maintain buffer size
    if (_landmarkBuffer.length > TARGET_FRAMES) {
      _landmarkBuffer.removeAt(0);
    }
    
    // Only predict if we have enough frames
    if (_landmarkBuffer.length < 10) return null; // Min frames needed
    
    // Aggregate features
    var features = _aggregateFeatures();
    
    // Apply scaling if scaler is loaded
    features = _scaleFeatures(features);
    
    // Run inference
    return _predict(features);
  }
  
  /// Apply StandardScaler transformation: (x - mean) / std
  List<double> _scaleFeatures(List<double> features) {
    if (_scalerMean == null || _scalerStd == null) {
      return features; // No scaling if scaler not loaded
    }
    
    return List.generate(features.length, (i) {
      double std = _scalerStd![i];
      if (std == 0) std = 1.0; // Avoid division by zero
      return (features[i] - _scalerMean![i]) / std;
    });
  }
  
  List<double> _aggregateFeatures() {
    // Shape: [frames, 225] -> [675] (mean, max, std)
    
    final int frames = _landmarkBuffer.length;
    final int dim = LANDMARKS_PER_FRAME;
    
    final mean = List<double>.filled(dim, 0.0);
    final maxVals = List<double>.filled(dim, -999.0);
    final std = List<double>.filled(dim, 0.0);
    
    // 1. Calculate Mean and Max
    for (int i = 0; i < frames; i++) {
      for (int j = 0; j < dim; j++) {
        double val = _landmarkBuffer[i][j];
        mean[j] += val;
        maxVals[j] = max(maxVals[j], val);
      }
    }
    
    for (int j = 0; j < dim; j++) {
      mean[j] /= frames;
    }
    
    // 2. Calculate Std Dev
    for (int i = 0; i < frames; i++) {
      for (int j = 0; j < dim; j++) {
        double val = _landmarkBuffer[i][j];
        double diff = val - mean[j];
        std[j] += diff * diff;
      }
    }
    
    for (int j = 0; j < dim; j++) {
      std[j] = sqrt(std[j] / frames);
    }
    
    // Concatenate [mean, max, std]
    return [...mean, ...maxVals, ...std];
  }
  
  Map<String, dynamic> _predict(List<double> features) {
    // Input tensor: [1, 675]
    var input = [features];
    
    // Output tensor: [1, num_classes]
    var output = List.filled(1 * _labels.length, 0.0).reshape([1, _labels.length]);
    
    _interpreter!.run(input, output);
    
    // Find best class
    List<double> probabilities = List<double>.from(output[0]);
    
    double maxScore = -1.0;
    int maxIndex = -1;
    
    for (int i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxScore) {
        maxScore = probabilities[i];
        maxIndex = i;
      }
    }
    
    String label = maxIndex >= 0 && maxIndex < _labels.length 
        ? _labels[maxIndex] 
        : 'Unknown';
        
    return {
      'label': label,
      'confidence': maxScore,
      'index': maxIndex
    };
  }
  
  void reset() {
    _landmarkBuffer.clear();
  }
  
  void close() {
    _interpreter?.close();
  }
}
