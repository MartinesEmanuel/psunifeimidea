# Diagnóstico Acústico Mobile

MVP Flutter para técnicos consultarem o sistema de diagnóstico acústico em campo.

## Escopo do MVP

- Seleção do modelo da máquina;
- envio de áudio ou vídeo e laudo de inspeção;
- relatório de hipótese, nível de confiança, peças prioritárias e possíveis causas;
- histórico local da sessão;
- interface sem recursos administrativos ou treinamento da IA.

A análise usa uma implementação simulada (`MockDiagnosisService`) para tornar o fluxo demonstrável sem expor a base de treinamento. A integração de produção deve substituir esse serviço por uma API autenticada do backend Python, sem expor o modo Admin no aplicativo.

## Executar

```bash
flutter pub get
flutter run
```

Requer Flutter 3.19+ e Dart 3.3+.

## APK de release

```bash
flutter build apk --release
```

O APK será gerado em `build/app/outputs/flutter-apk/app-release.apk`.
