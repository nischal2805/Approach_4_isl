import 'dart:io';
import 'dart:async';
import 'package:flutter/services.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:path_provider/path_provider.dart';

/// Service to handle grammar correction using on-device LLM (SmolLM2)
/// with comprehensive rule-based fallback.
class GrammarService {
  LlamaProcessor? _llamaProcessor;
  bool _isModelLoaded = false;
  
  // Conversation history for context
  final List<String> _conversationHistory = [];
  static const int MAX_HISTORY = 5;
  
  // LLM settings
  static const Duration LLM_TIMEOUT = Duration(seconds: 5);
  static const int MAX_TOKENS = 50;
  
  /// Initialize the service - LLM is default, rule-based is fallback
  Future<void> initialize({bool useLLM = true}) async {
    if (useLLM) {
      try {
        final modelPath = await _copyModelToLocal();
        if (modelPath != null && await File(modelPath).exists()) {
          _llamaProcessor = LlamaProcessor(modelPath);
          _isModelLoaded = true;
          print("GrammarService: LLM loaded successfully");
        } else {
          print("GrammarService: Model file not found, using rules");
        }
      } catch (e) {
        print("GrammarService: LLM init failed ($e), using rules");
      }
    }
  }

  /// Main correction method - tries LLM first, falls back to rules
  Future<String> correctGrammar(String brokenSentence) async {
    if (brokenSentence.trim().isEmpty) return "";
    
    String corrected;
    
    if (_isModelLoaded) {
      // Try LLM with timeout
      try {
        corrected = await _correctWithLLM(brokenSentence)
            .timeout(LLM_TIMEOUT, onTimeout: () {
          print("GrammarService: LLM timeout, using rules");
          return _correctWithRules(brokenSentence);
        });
      } catch (e) {
        print("GrammarService: LLM error ($e), using rules");
        corrected = _correctWithRules(brokenSentence);
      }
    } else {
      corrected = _correctWithRules(brokenSentence);
    }
    
    // Add to history for context
    _conversationHistory.add(corrected);
    if (_conversationHistory.length > MAX_HISTORY) {
      _conversationHistory.removeAt(0);
    }
    
    return corrected;
  }

  Future<String> _correctWithLLM(String input) async {
    // Build context from history
    String context = "";
    if (_conversationHistory.isNotEmpty) {
      context = "Context: ${_conversationHistory.takeLast(3).join(' ')}\n";
    }
    
    final prompt = """<|im_start|>system
You convert Sign Language gloss to grammatical English. Output only the corrected sentence, nothing else.
<|im_end|>
<|im_start|>user
${context}Gloss: $input
English:<|im_end|>
<|im_start|>assistant
""";

    StringBuffer response = StringBuffer();
    int tokenCount = 0;
    
    await for (final token in _llamaProcessor!.stream(prompt)) {
      response.write(token);
      tokenCount++;
      if (tokenCount >= MAX_TOKENS) break;
      if (token.contains('<|im_end|>')) break;
    }
    
    String result = response.toString()
        .replaceAll('<|im_end|>', '')
        .replaceAll('<|im_start|>', '')
        .trim();
    
    // If LLM returns empty or garbage, use rules
    if (result.isEmpty || result.length < 2) {
      return _correctWithRules(input);
    }
    
    return result;
  }

  /// Comprehensive ISL-to-English rule-based correction
  String _correctWithRules(String input) {
    if (input.trim().isEmpty) return "";
    
    String text = input.trim();
    List<String> words = text.split(RegExp(r'\s+'));
    
    // === STEP 1: Word-level replacements (ISL glosses) ===
    final Map<String, String> glossMap = {
      // Pronouns
      'ME': 'I', 'MY': 'my', 'MINE': 'mine',
      'YOU': 'you', 'YOUR': 'your',
      'HE': 'he', 'SHE': 'she', 'IT': 'it',
      'WE': 'we', 'THEY': 'they',
      
      // Common verbs (ISL often uses base form)
      'GO': 'go', 'GOING': 'going', 'WENT': 'went',
      'COME': 'come', 'COMING': 'coming',
      'EAT': 'eat', 'EATING': 'eating',
      'DRINK': 'drink', 'DRINKING': 'drinking',
      'WANT': 'want', 'NEED': 'need',
      'LIKE': 'like', 'LOVE': 'love',
      'HELP': 'help', 'WORK': 'work',
      'LEARN': 'learn', 'STUDY': 'study',
      'UNDERSTAND': 'understand',
      
      // Question words
      'WHAT': 'what', 'WHERE': 'where', 'WHEN': 'when',
      'WHY': 'why', 'HOW': 'how', 'WHO': 'who',
      
      // Time markers
      'NOW': 'now', 'TODAY': 'today', 'TOMORROW': 'tomorrow',
      'YESTERDAY': 'yesterday', 'LATER': 'later',
      'BEFORE': 'before', 'AFTER': 'after',
      
      // Common nouns
      'SCHOOL': 'school', 'HOME': 'home', 'WORK': 'work',
      'FOOD': 'food', 'WATER': 'water', 'MONEY': 'money',
      'FRIEND': 'friend', 'FAMILY': 'family',
      'MOTHER': 'mother', 'FATHER': 'father',
      'BROTHER': 'brother', 'SISTER': 'sister',
      
      // Greetings
      'HELLO': 'Hello', 'HI': 'Hi', 'BYE': 'Goodbye',
      'THANK': 'Thank you', 'SORRY': 'Sorry',
      'PLEASE': 'please', 'YES': 'yes', 'NO': 'no',
      'GOOD': 'good', 'BAD': 'bad',
      'MORNING': 'morning', 'AFTERNOON': 'afternoon',
      'EVENING': 'evening', 'NIGHT': 'night',
    };
    
    // Apply word replacements
    words = words.map((w) {
      String upper = w.toUpperCase();
      return glossMap[upper] ?? w.toLowerCase();
    }).toList();
    
    // === STEP 2: Pattern-based corrections ===
    String sentence = words.join(' ');
    
    // Common ISL patterns -> English
    final patterns = [
      // "ME GO SCHOOL" -> "I am going to school"
      (RegExp(r'\bi go (\w+)'), 'I am going to \$1'),
      (RegExp(r'\bi going (\w+)'), 'I am going to \$1'),
      
      // "YOU HOW" -> "How are you?"
      (RegExp(r'you how\b'), 'How are you?'),
      (RegExp(r'how you\b'), 'How are you?'),
      
      // "WHAT NAME YOU" -> "What is your name?"
      (RegExp(r'what name you'), 'What is your name?'),
      (RegExp(r'name you what'), 'What is your name?'),
      
      // "WHERE YOU GO" -> "Where are you going?"
      (RegExp(r'where you go'), 'Where are you going?'),
      (RegExp(r'you go where'), 'Where are you going?'),
      
      // "I WANT FOOD" -> "I want food"
      (RegExp(r'\bi want (\w+)'), 'I want \$1'),
      
      // "HE/SHE GO" -> "He/She is going"
      (RegExp(r'\b(he|she) go\b'), '\$1 is going'),
      
      // "GOOD MORNING" -> "Good morning"
      (RegExp(r'good morning'), 'Good morning'),
      (RegExp(r'good afternoon'), 'Good afternoon'),
      (RegExp(r'good evening'), 'Good evening'),
      (RegExp(r'good night'), 'Good night'),
      
      // Add articles where missing
      (RegExp(r'\bgo to (\w+)\b(?!\s+(?:a|an|the))'), 'go to the \$1'),
    ];
    
    for (var pattern in patterns) {
      sentence = sentence.replaceAllMapped(pattern.$1, (m) {
        String replacement = pattern.$2;
        for (int i = 0; i <= m.groupCount; i++) {
          replacement = replacement.replaceAll('\$$i', m.group(i) ?? '');
        }
        return replacement;
      });
    }
    
    // === STEP 3: Final cleanup ===
    // Capitalize first letter
    if (sentence.isNotEmpty) {
      sentence = sentence[0].toUpperCase() + sentence.substring(1);
    }
    
    // Add punctuation if missing
    if (!sentence.endsWith('.') && !sentence.endsWith('?') && !sentence.endsWith('!')) {
      // Add ? for questions
      if (sentence.toLowerCase().startsWith('what') ||
          sentence.toLowerCase().startsWith('where') ||
          sentence.toLowerCase().startsWith('when') ||
          sentence.toLowerCase().startsWith('why') ||
          sentence.toLowerCase().startsWith('how') ||
          sentence.toLowerCase().startsWith('who')) {
        sentence += '?';
      } else {
        sentence += '.';
      }
    }
    
    // Fix double spaces
    sentence = sentence.replaceAll(RegExp(r'\s+'), ' ').trim();
    
    return sentence;
  }
  
  Future<String?> _copyModelToLocal() async {
    try {
      // Use app documents directory for persistence
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/smollm2.gguf');
      
      // Check if already copied
      if (await file.exists()) {
        return file.path;
      }
      
      // Copy from assets
      final data = await rootBundle.load('assets/smollm2-135m-q4_k_m.gguf');
      await file.writeAsBytes(data.buffer.asUint8List(), flush: true);
      return file.path;
    } catch (e) {
      print("Failed to copy model: $e");
      return null;
    }
  }
  
  /// Clear conversation history
  void clearHistory() {
    _conversationHistory.clear();
  }
  
  void dispose() {
    _llamaProcessor?.unloadModel();
  }
}

// Extension for List.takeLast
extension TakeLast<T> on List<T> {
  List<T> takeLast(int n) {
    if (n >= length) return this;
    return sublist(length - n);
  }
}
