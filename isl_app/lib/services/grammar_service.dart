import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:llama_sdk/llama_sdk.dart';
import '../constants.dart';

/// Service to handle grammar correction for ISL gloss → English
/// Uses SmolLM2 LLM via llama_sdk for natural language generation
class GrammarService {
  Llama? _llama;
  bool _isInitialized = false;
  bool _useLLM = true;
  bool _isGenerating = false;
  
  /// Check if LLM is available
  bool get isLLMAvailable => _isInitialized && _useLLM;

  /// Initialize the LLM model
  Future<void> initialize({bool useLLM = true}) async {
    _useLLM = useLLM;
    
    if (!_useLLM) {
      if (kDebugMode) debugPrint("GrammarService: Using rule-based correction (LLM disabled)");
      _isInitialized = true;
      return;
    }
    
    try {
      if (kDebugMode) debugPrint("GrammarService: Loading LLM model with llama_sdk...");
      
      // Copy model from assets to app directory
      final modelPath = await _copyAssetToFile(AssetPaths.llmModel);
      if (kDebugMode) debugPrint("GrammarService: Model path: $modelPath");
      
      final modelFile = File(modelPath);
      if (!await modelFile.exists()) {
        throw Exception("Model file not found at $modelPath");
      }
      
      // Initialize llama_sdk with optimal settings for mobile
      // Uses LlamaController (not LlamaParams) with modelPath string
      _llama = Llama(LlamaController(
        modelPath: modelPath,
        nCtx: LLMSettings.contextSize,
        nBatch: LLMSettings.batchSize,
        greedy: true, // Deterministic output for grammar correction
      ));
      
      _isInitialized = true;
      if (kDebugMode) debugPrint("GrammarService: LLM initialized successfully!");
    } catch (e) {
      if (kDebugMode) debugPrint("GrammarService: LLM initialization failed: $e");
      if (kDebugMode) debugPrint("GrammarService: Falling back to rule-based correction");
      _useLLM = false;
      _isInitialized = true;
    }
  }

  /// Copy asset file to app documents directory for native access
  Future<String> _copyAssetToFile(String assetPath) async {
    final appDir = await getApplicationDocumentsDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${appDir.path}/$fileName');
    
    // Check if already copied
    if (await file.exists()) {
      final fileSize = await file.length();
      if (fileSize > 1000000) { // > 1MB means it's a real model file
        if (kDebugMode) debugPrint("GrammarService: Model already cached (${(fileSize / 1024 / 1024).toStringAsFixed(1)} MB)");
        return file.path;
      }
    }
    
    // Copy from assets
    if (kDebugMode) debugPrint("GrammarService: Copying model to cache...");
    try {
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List());
      if (kDebugMode) debugPrint("GrammarService: Model copied (${(data.lengthInBytes / 1024 / 1024).toStringAsFixed(1)} MB)");
    } catch (e) {
      // Try smaller model if main model not found
      if (kDebugMode) debugPrint("GrammarService: Main model not found, trying smaller model...");
      final smallData = await rootBundle.load(AssetPaths.llmModelSmall);
      final smallFile = File('${appDir.path}/${AssetPaths.llmModelSmall.split('/').last}');
      await smallFile.writeAsBytes(smallData.buffer.asUint8List());
      return smallFile.path;
    }
    
    return file.path;
  }

  /// Main correction method - uses LLM or falls back to rules
  Future<String> correctGrammar(String glossInput) async {
    if (kDebugMode) debugPrint('GrammarService: Input gloss: "$glossInput"');
    
    if (glossInput.trim().isEmpty) return "";
    
    String result;
    
    if (_useLLM && _llama != null && !_isGenerating) {
      result = await _correctWithLLM(glossInput);
    } else {
      result = _correctWithRules(glossInput);
    }
    
    if (kDebugMode) debugPrint('GrammarService: Output: "$result"');
    
    return result;
  }

  /// Use LLM for grammar correction
  Future<String> _correctWithLLM(String glossInput) async {
    if (_llama == null) return _correctWithRules(glossInput);
    
    _isGenerating = true;
    
    try {
      final prompt = _buildPrompt(glossInput);
      if (kDebugMode) debugPrint('GrammarService: Sending to LLM...');
      
      // Collect response tokens
      final buffer = StringBuffer();
      
      // Stream tokens from llama_sdk
      await for (final token in _llama!.prompt([
        LlamaMessage.withRole(role: 'user', content: prompt),
      ])) {
        // Check for stop conditions
        if (token.contains('<|im_end|>') || 
            token.contains('\n') ||
            buffer.length > 200) {
          break;
        }
        buffer.write(token);
        
        // Check for sentence end
        final text = buffer.toString();
        if (text.length > 10 && 
            (text.endsWith('.') || text.endsWith('!') || text.endsWith('?'))) {
          break;
        }
      }
      
      // Clean up response
      final response = _cleanLLMResponse(buffer.toString(), glossInput);
      
      _isGenerating = false;
      
      if (response.isNotEmpty) {
        return response;
      }
    } catch (e) {
      if (kDebugMode) debugPrint('GrammarService: LLM error: $e');
      _isGenerating = false;
    }
    
    // Fallback to rules if LLM fails
    return _correctWithRules(glossInput);
  }

  /// Build prompt for grammar correction
  String _buildPrompt(String glossInput) {
    return '''<|im_start|>system
You are a grammar correction assistant. Convert Indian Sign Language (ISL) gloss notation into proper English sentences. ISL gloss uses simplified word order without articles and conjugation. Output ONLY the corrected sentence.
<|im_end|>
<|im_start|>user
Convert this ISL gloss to proper English: "$glossInput"
<|im_end|>
<|im_start|>assistant
''';
  }

  /// Clean up LLM response
  String _cleanLLMResponse(String response, String originalGloss) {
    var cleaned = response
        .replaceAll('<|im_start|>', '')
        .replaceAll('<|im_end|>', '')
        .replaceAll('assistant', '')
        .replaceAll(RegExp(r'[\n\r]+'), ' ')
        .trim();
    
    // Remove common artifacts
    final artifactPatterns = [
      RegExp(r'^(Here is|The sentence is|Corrected:?)[\s:]*', caseSensitive: false),
      RegExp(r'^"(.+)"$'),
      RegExp(r"^'(.+)'$"),
    ];
    
    for (final pattern in artifactPatterns) {
      final match = pattern.firstMatch(cleaned);
      if (match != null && match.groupCount >= 1) {
        cleaned = match.group(1) ?? cleaned;
      } else {
        cleaned = cleaned.replaceAll(pattern, '');
      }
    }
    
    cleaned = cleaned.trim();
    
    // Capitalize first letter
    if (cleaned.isNotEmpty) {
      cleaned = cleaned[0].toUpperCase() + cleaned.substring(1);
    }
    
    // Add period if missing
    if (cleaned.isNotEmpty && !cleaned.endsWith('.') && !cleaned.endsWith('!') && !cleaned.endsWith('?')) {
      cleaned = '$cleaned.';
    }
    
    return cleaned.isEmpty ? _correctWithRules(originalGloss) : cleaned;
  }

  /// Rule-based fallback for grammar correction
  String _correctWithRules(String glossInput) {
    if (glossInput.trim().isEmpty) return "";
    
    final words = glossInput.split(RegExp(r'\s+'));
    
    if (words.isEmpty) return "";
    if (words.length == 1) {
      return _capitalizeFirst(words[0].toLowerCase());
    }
    
    // Apply ISL → English transformations
    var result = words.map((w) => w.toLowerCase()).toList();
    
    // Word mappings for common ISL signs
    final wordMappings = {
      'i': 'I',
      'me': 'I',
      'my': 'my',
      'you': 'you',
      'your': 'your',
      'he': 'he',
      'she': 'she',
      'we': 'we',
      'they': 'they',
      'what': 'What',
      'where': 'Where',
      'when': 'When',
      'why': 'Why',
      'how': 'How',
      'who': 'Who',
    };
    
    result = result.map((w) => wordMappings[w] ?? w).toList();
    
    // Add articles where needed
    final needsArticle = {'school', 'hospital', 'home', 'market', 'office', 'store', 'book', 'car', 'bus', 'train'};
    for (var i = result.length - 1; i >= 0; i--) {
      if (needsArticle.contains(result[i]) && 
          (i == 0 || !['the', 'a', 'an', 'my', 'your', 'his', 'her', 'their', 'our'].contains(result[i-1]))) {
        final vowels = ['a', 'e', 'i', 'o', 'u'];
        final article = vowels.contains(result[i][0]) ? 'an' : 'a';
        result.insert(i, article);
      }
    }
    
    // Add "to" before verbs after go/went
    for (var i = 0; i < result.length - 1; i++) {
      if ((result[i] == 'go' || result[i] == 'went' || result[i] == 'going') &&
          !['to', 'home'].contains(result[i + 1])) {
        result.insert(i + 1, 'to');
        break;
      }
    }
    
    // Handle tense markers
    if (result.contains('yesterday') || result.contains('past')) {
      result = result.map((w) {
        switch (w) {
          case 'go': return 'went';
          case 'eat': return 'ate';
          case 'come': return 'came';
          case 'see': return 'saw';
          case 'do': return 'did';
          case 'have': return 'had';
          case 'is': return 'was';
          case 'are': return 'were';
          default: return w;
        }
      }).toList();
    }
    
    if (result.contains('tomorrow') || result.contains('future')) {
      final verbIdx = result.indexWhere((w) => 
        ['go', 'eat', 'come', 'see', 'do', 'have'].contains(w));
      if (verbIdx > 0) {
        result.insert(verbIdx, 'will');
      }
    }
    
    // Build final sentence
    var sentence = result.join(' ');
    
    // Clean up
    sentence = sentence.replaceAll(RegExp(r'\s+'), ' ').trim();
    
    // Capitalize and add period
    if (sentence.isNotEmpty) {
      sentence = sentence[0].toUpperCase() + sentence.substring(1);
      if (!sentence.endsWith('.') && !sentence.endsWith('!') && !sentence.endsWith('?')) {
        sentence = '$sentence.';
      }
    }
    
    return sentence;
  }

  String _capitalizeFirst(String s) {
    if (s.isEmpty) return s;
    return s[0].toUpperCase() + s.substring(1);
  }

  /// Dispose resources
  void dispose() {
    // llama_sdk uses reload() to free resources, not dispose()
    _llama?.reload();
    _llama = null;
    _isInitialized = false;
  }
}
