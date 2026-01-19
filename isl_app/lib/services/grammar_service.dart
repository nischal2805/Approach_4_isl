import 'package:flutter/foundation.dart';

/// Service to handle grammar correction for ISL
/// Uses rule-based approach for reliable offline operation
class GrammarService {
  // Conversation history for context
  final List<String> _conversationHistory = [];
  static const int maxHistory = 5;

  /// Initialize the service - uses rule-based grammar correction
  Future<void> initialize({bool useLLM = false}) async {
    // Using robust rule-based approach for offline reliability
    debugPrint("GrammarService: Initialized with rule-based correction");
  }

  /// Main correction method - uses rule-based approach
  Future<String> correctGrammar(String brokenSentence) async {
    if (brokenSentence.trim().isEmpty) return "";

    // Always use rule-based correction for reliability
    String corrected = _correctWithRules(brokenSentence);

    // Add to history for context
    _conversationHistory.add(corrected);
    if (_conversationHistory.length > maxHistory) {
      _conversationHistory.removeAt(0);
    }

    return corrected;
  }

  /// Comprehensive ISL-to-English rule-based correction
  String _correctWithRules(String input) {
    if (input.trim().isEmpty) return "";

    String text = input.trim();

    // Split into words
    List<String> words =
        text.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();
    if (words.isEmpty) return "";

    // 1. Subject pronoun fixes
    words = _fixPronouns(words);

    // 2. Handle common ISL patterns
    String sentence = _handlePatterns(words);

    // 3. Apply grammar fixes
    sentence = _applyGrammarFixes(sentence);

    // 4. Capitalize and punctuate
    sentence = _formatSentence(sentence);

    return sentence;
  }

  List<String> _fixPronouns(List<String> words) {
    final pronounMap = {
      'ME': 'I',
      'MY': 'my',
      'YOU': 'you',
      'YOUR': 'your',
      'HE': 'he',
      'SHE': 'she',
      'WE': 'we',
      'THEY': 'they',
      'SELF': 'myself',
    };

    return words.map((w) => pronounMap[w.toUpperCase()] ?? w).toList();
  }

  String _handlePatterns(List<String> words) {
    String joinedLower = words.join(" ").toLowerCase();

    // Pattern: PERSON + ACTION → "The person is doing action"
    // Pattern: PLACE + GO → "Going to place"
    // Pattern: OBJECT + WANT → "I want object"

    // Check for common greetings (preserve as-is)
    if (_isGreeting(joinedLower)) {
      return _formatGreeting(words);
    }

    // Check for questions
    if (_isQuestion(words)) {
      return _formQuestion(words);
    }

    // Default: reconstruct with basic grammar
    return words.join(" ").toLowerCase();
  }

  bool _isGreeting(String text) {
    final greetings = [
      'hello',
      'hi',
      'good morning',
      'good afternoon',
      'good evening',
      'good night',
      'thank you',
      'thanks',
      'welcome',
      'bye',
      'goodbye'
    ];
    return greetings.any((g) => text.contains(g));
  }

  String _formatGreeting(List<String> words) {
    String joined = words.join(" ").toLowerCase();
    // Capitalize first letter of each major word
    return words.map((w) {
      String lower = w.toLowerCase();
      if ([
        'good',
        'thank',
        'hello',
        'hi',
        'bye',
        'you',
        'morning',
        'afternoon',
        'evening',
        'night',
        'welcome'
      ].contains(lower)) {
        return w[0].toUpperCase() + w.substring(1).toLowerCase();
      }
      return lower;
    }).join(" ");
  }

  bool _isQuestion(List<String> words) {
    final questionWords = [
      'what',
      'where',
      'when',
      'who',
      'why',
      'how',
      'which'
    ];
    return words.any((w) => questionWords.contains(w.toLowerCase()));
  }

  String _formQuestion(List<String> words) {
    // Move question word to front
    final questionWords = [
      'what',
      'where',
      'when',
      'who',
      'why',
      'how',
      'which'
    ];
    String? qWord;
    List<String> others = [];

    for (var w in words) {
      if (questionWords.contains(w.toLowerCase()) && qWord == null) {
        qWord = w.toLowerCase();
      } else {
        others.add(w.toLowerCase());
      }
    }

    if (qWord != null) {
      String qWordCapitalized = qWord[0].toUpperCase() + qWord.substring(1);
      if (others.isEmpty) {
        return "$qWordCapitalized?";
      }
      return "$qWordCapitalized ${others.join(" ")}?";
    }

    return words.join(" ").toLowerCase();
  }

  String _applyGrammarFixes(String sentence) {
    // Add "I am" before single verbs
    if (RegExp(r'^(go|want|need|like|love|have|see|know)\b',
            caseSensitive: false)
        .hasMatch(sentence)) {
      sentence = "I $sentence";
    }

    // Fix common verb patterns
    sentence = sentence.replaceAllMapped(
      RegExp(r'\bI go\b', caseSensitive: false),
      (m) => "I am going",
    );
    sentence = sentence.replaceAllMapped(
      RegExp(r'\bI want\b', caseSensitive: false),
      (m) => "I want",
    );

    // Add articles before common nouns
    final nounsNeedingArticle = [
      'school',
      'hospital',
      'doctor',
      'teacher',
      'bus',
      'train',
      'car',
      'house',
      'shop'
    ];
    for (var noun in nounsNeedingArticle) {
      sentence = sentence.replaceAllMapped(
        RegExp('\\b$noun\\b', caseSensitive: false),
        (m) => "the $noun",
      );
    }
    // Clean up double articles
    sentence = sentence.replaceAll(RegExp(r'\bthe the\b'), 'the');

    return sentence;
  }

  String _formatSentence(String sentence) {
    if (sentence.isEmpty) return "";

    // Capitalize first letter
    sentence = sentence[0].toUpperCase() + sentence.substring(1);

    // Add period if no punctuation
    if (!sentence.endsWith('.') &&
        !sentence.endsWith('!') &&
        !sentence.endsWith('?')) {
      sentence += '.';
    }

    return sentence;
  }

  /// Clear conversation history
  void clearHistory() {
    _conversationHistory.clear();
  }

  /// Dispose of resources
  void dispose() {
    _conversationHistory.clear();
  }
}
