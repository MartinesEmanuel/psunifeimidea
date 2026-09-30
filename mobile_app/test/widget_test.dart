import 'package:diagnostico_acustico_mobile/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('exibe a tela de nova inspeção', (tester) async {
    await tester.pumpWidget(const DiagnosticApp());
    expect(find.text('Nova inspeção'), findsOneWidget);
    expect(find.text('Gerar diagnóstico'), findsOneWidget);
  });
}
