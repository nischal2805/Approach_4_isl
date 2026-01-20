// Basic widget test for ISL Translator
import 'package:flutter_test/flutter_test.dart';
import 'package:isl_translator/main.dart';

void main() {
  testWidgets('App launches correctly', (WidgetTester tester) async {
    await tester.pumpWidget(const ISLTranslatorApp());

    // Verify that the app has navigation destinations
    expect(find.text('Sign → Text'), findsOneWidget);
    expect(find.text('Text → Sign'), findsOneWidget);
  });
}
