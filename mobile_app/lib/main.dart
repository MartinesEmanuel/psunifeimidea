import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() => runApp(const DiagnosticApp());

class AppColors {
  static const ink = Color(0xFF07111E);
  static const panel = Color(0xFF101F30);
  static const panelSoft = Color(0xFF16293D);
  static const line = Color(0xFF294158);
  static const cyan = Color(0xFF37D6C0);
  static const amber = Color(0xFFFFB85C);
  static const coral = Color(0xFFFF6B5A);
  static const text = Color(0xFFF4F8FC);
  static const muted = Color(0xFF9DB0C2);
}

class DiagnosticApp extends StatefulWidget {
  const DiagnosticApp({super.key});

  @override
  State<DiagnosticApp> createState() => _DiagnosticAppState();
}

class _DiagnosticAppState extends State<DiagnosticApp> {
  final List<DiagnosisReport> _history = [];

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Midea Diagnose',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: AppColors.ink,
        colorScheme: const ColorScheme.dark(
          primary: AppColors.cyan,
          secondary: AppColors.amber,
          surface: AppColors.panel,
          onSurface: AppColors.text,
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: AppColors.panel,
          labelStyle: const TextStyle(color: AppColors.muted),
          hintStyle: const TextStyle(color: AppColors.muted),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: AppColors.line),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: AppColors.line),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: AppColors.cyan, width: 1.5),
          ),
        ),
      ),
      home: AppShell(
        history: _history,
        onNewReport: (report) => setState(() => _history.insert(0, report)),
      ),
    );
  }
}

class AppShell extends StatefulWidget {
  const AppShell({super.key, required this.history, required this.onNewReport});

  final List<DiagnosisReport> history;
  final ValueChanged<DiagnosisReport> onNewReport;

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _index = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      DiagnosePage(onNewReport: widget.onNewReport),
      HistoryPage(history: widget.history),
    ];
    return Scaffold(
      body: SafeArea(child: pages[_index]),
      bottomNavigationBar: NavigationBar(
        backgroundColor: AppColors.panel,
        indicatorColor: AppColors.panelSoft,
        selectedIndex: _index,
        onDestinationSelected: (value) => setState(() => _index = value),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.graphic_eq_outlined), selectedIcon: Icon(Icons.graphic_eq), label: 'Diagnosticar'),
          NavigationDestination(icon: Icon(Icons.history_outlined), selectedIcon: Icon(Icons.history), label: 'Histórico'),
        ],
      ),
    );
  }
}

class DiagnosePage extends StatefulWidget {
  const DiagnosePage({super.key, required this.onNewReport});

  final ValueChanged<DiagnosisReport> onNewReport;

  @override
  State<DiagnosePage> createState() => _DiagnosePageState();
}

class _DiagnosePageState extends State<DiagnosePage> {
  final _formKey = GlobalKey<FormState>();
  final _modelController = TextEditingController();
  final _reportController = TextEditingController();
  final _service = MockDiagnosisService();
  PlatformFile? _media;
  String _condition = 'Em operação';
  bool _loading = false;

  @override
  void dispose() {
    _modelController.dispose();
    _reportController.dispose();
    super.dispose();
  }

  Future<void> _selectMedia() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['mp4', 'mov', 'm4a', 'mp3', 'wav', 'aac', 'ogg', 'flac'],
    );
    if (result != null && result.files.isNotEmpty) {
      setState(() => _media = result.files.single);
    }
  }

  Future<void> _analyze() async {
    if (!_formKey.currentState!.validate() || _media == null) {
      if (_media == null) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Selecione um áudio ou vídeo para analisar.')));
      }
      return;
    }
    setState(() => _loading = true);
    final report = await _service.analyze(
      machineModel: _modelController.text.trim(),
      mediaName: _media!.name,
      operatorReport: _reportController.text.trim(),
      condition: _condition,
    );
    if (!mounted) return;
    setState(() => _loading = false);
    widget.onNewReport(report);
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => ReportPage(report: report)));
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 32),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const BrandHeader(),
            const SizedBox(height: 28),
            const Text('Nova inspeção', style: TextStyle(color: AppColors.text, fontSize: 27, fontWeight: FontWeight.w700, letterSpacing: -0.8)),
            const SizedBox(height: 6),
            const Text('Registre o contexto e envie a evidência sonora da máquina.', style: TextStyle(color: AppColors.muted, height: 1.4)),
            const SizedBox(height: 24),
            TextFormField(
              controller: _modelController,
              textCapitalization: TextCapitalization.characters,
              decoration: const InputDecoration(labelText: 'Modelo da máquina', hintText: 'Ex.: LAC12X / família da unidade', prefixIcon: Icon(Icons.precision_manufacturing_outlined)),
              validator: (value) => value == null || value.trim().isEmpty ? 'Informe o modelo da máquina.' : null,
            ),
            const SizedBox(height: 16),
            const FieldLabel(label: 'Condição no momento da gravação'),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: ['Em operação', 'Partida', 'Carga elevada', 'Parada'].map((option) => ChoiceChip(
                label: Text(option),
                selected: _condition == option,
                onSelected: (_) => setState(() => _condition = option),
                selectedColor: AppColors.cyan.withValues(alpha: .20),
                backgroundColor: AppColors.panel,
                side: BorderSide(color: _condition == option ? AppColors.cyan : AppColors.line),
              )).toList(),
            ),
            const SizedBox(height: 20),
            const FieldLabel(label: 'Evidência sonora'),
            const SizedBox(height: 8),
            InkWell(
              onTap: _selectMedia,
              borderRadius: BorderRadius.circular(18),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(color: AppColors.panel, borderRadius: BorderRadius.circular(18), border: Border.all(color: _media == null ? AppColors.line : AppColors.cyan)),
                child: Row(
                  children: [
                    Container(
                      width: 48,
                      height: 48,
                      decoration: BoxDecoration(color: AppColors.panelSoft, borderRadius: BorderRadius.circular(14)),
                      child: Icon(_media == null ? Icons.upload_file_outlined : Icons.audio_file_outlined, color: _media == null ? AppColors.muted : AppColors.cyan),
                    ),
                    const SizedBox(width: 14),
                    Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(_media?.name ?? 'Adicionar áudio ou vídeo', style: const TextStyle(fontWeight: FontWeight.w600)),
                      const SizedBox(height: 3),
                      Text(_media == null ? 'MP4, MOV, WAV, MP3 e outros formatos aceitos' : '${(_media!.size / 1024 / 1024).toStringAsFixed(1)} MB · toque para trocar', style: const TextStyle(color: AppColors.muted, fontSize: 12)),
                    ])),
                    const Icon(Icons.chevron_right, color: AppColors.muted),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            TextFormField(
              controller: _reportController,
              minLines: 4,
              maxLines: 6,
              decoration: const InputDecoration(labelText: 'Laudo do técnico', hintText: 'Descreva ruído, local, sintomas e condição observada.'),
              validator: (value) => value == null || value.trim().length < 12 ? 'Descreva a ocorrência com pelo menos 12 caracteres.' : null,
            ),
            const SizedBox(height: 16),
            const SafetyNote(),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              height: 54,
              child: FilledButton.icon(
                onPressed: _loading ? null : _analyze,
                style: FilledButton.styleFrom(backgroundColor: AppColors.cyan, foregroundColor: AppColors.ink, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                icon: _loading ? const SizedBox(width: 19, height: 19, child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.ink)) : const Icon(Icons.auto_awesome),
                label: Text(_loading ? 'Analisando assinatura...' : 'Gerar diagnóstico', style: const TextStyle(fontWeight: FontWeight.w800)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ReportPage extends StatelessWidget {
  const ReportPage({super.key, required this.report});
  final DiagnosisReport report;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(backgroundColor: AppColors.ink, foregroundColor: AppColors.text, title: const Text('Relatório da inspeção')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 6, 20, 32),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          StatusBadge(level: report.confidenceLevel),
          const SizedBox(height: 16),
          Text(report.mainHypothesis, style: const TextStyle(fontSize: 31, fontWeight: FontWeight.w700, letterSpacing: -1)),
          const SizedBox(height: 8),
          Text('${report.machineModel} · ${report.condition}', style: const TextStyle(color: AppColors.muted)),
          const SizedBox(height: 22),
          ConfidenceGauge(value: report.confidence),
          const SizedBox(height: 10),
          Text(report.guidance, style: const TextStyle(color: AppColors.muted, height: 1.45)),
          const SizedBox(height: 24),
          ReportSection(title: 'Peças ou conjuntos prioritários', icon: Icons.build_outlined, items: report.priorityParts),
          const SizedBox(height: 14),
          ReportSection(title: 'Possíveis origens do problema', icon: Icons.account_tree_outlined, items: report.possibleCauses),
          const SizedBox(height: 14),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: AppColors.panel, borderRadius: BorderRadius.circular(18), border: Border.all(color: AppColors.line)),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Text('Próximos passos', style: TextStyle(fontWeight: FontWeight.w700)),
              const SizedBox(height: 10),
              ...report.nextSteps.asMap().entries.map((entry) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text('${entry.key + 1}. ${entry.value}', style: const TextStyle(color: AppColors.muted, height: 1.35)),
              )),
            ]),
          ),
          const SizedBox(height: 18),
          OutlinedButton.icon(
            onPressed: () async {
              await Clipboard.setData(ClipboardData(text: report.asText));
              if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Relatório copiado para a área de transferência.')));
            },
            icon: const Icon(Icons.content_copy_outlined),
            label: const Text('Copiar relatório'),
            style: OutlinedButton.styleFrom(foregroundColor: AppColors.text, side: const BorderSide(color: AppColors.line), minimumSize: const Size.fromHeight(50)),
          ),
        ]),
      ),
    );
  }
}

class HistoryPage extends StatelessWidget {
  const HistoryPage({super.key, required this.history});
  final List<DiagnosisReport> history;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const BrandHeader(compact: true),
        const SizedBox(height: 28),
        const Text('Histórico desta sessão', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w700)),
        const SizedBox(height: 8),
        Text(history.isEmpty ? 'Os diagnósticos realizados neste aparelho aparecerão aqui.' : '${history.length} relatório(s) gerado(s) nesta sessão.', style: const TextStyle(color: AppColors.muted)),
        const SizedBox(height: 20),
        Expanded(child: history.isEmpty
            ? const Center(child: Icon(Icons.history_toggle_off_outlined, size: 58, color: AppColors.line))
            : ListView.separated(
                itemCount: history.length,
                separatorBuilder: (_, __) => const SizedBox(height: 10),
                itemBuilder: (_, index) {
                  final report = history[index];
                  return ListTile(
                    tileColor: AppColors.panel,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16), side: const BorderSide(color: AppColors.line)),
                    leading: CircleAvatar(backgroundColor: AppColors.cyan.withValues(alpha: .16), child: const Icon(Icons.graphic_eq, color: AppColors.cyan)),
                    title: Text(report.mainHypothesis),
                    subtitle: Text('${report.machineModel} · ${(report.confidence * 100).round()}%'),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => ReportPage(report: report))),
                  );
                },
              )),
      ]),
    );
  }
}

class BrandHeader extends StatelessWidget {
  const BrandHeader({super.key, this.compact = false});
  final bool compact;

  @override
  Widget build(BuildContext context) => Row(children: [
    Container(width: compact ? 38 : 44, height: compact ? 38 : 44, decoration: BoxDecoration(color: AppColors.cyan, borderRadius: BorderRadius.circular(13)), child: const Icon(Icons.multitrack_audio, color: AppColors.ink)),
    const SizedBox(width: 11),
    const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('MIDEA · FIELD DIAGNOSE', style: TextStyle(fontSize: 11, color: AppColors.cyan, fontWeight: FontWeight.w800, letterSpacing: 1.25)),
      Text('Diagnóstico acústico', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
    ]),
  ]);
}

class FieldLabel extends StatelessWidget {
  const FieldLabel({super.key, required this.label});
  final String label;
  @override
  Widget build(BuildContext context) => Text(label, style: const TextStyle(color: AppColors.muted, fontSize: 12, fontWeight: FontWeight.w700, letterSpacing: .35));
}

class SafetyNote extends StatelessWidget {
  const SafetyNote({super.key});
  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(color: AppColors.amber.withValues(alpha: .10), borderRadius: BorderRadius.circular(14), border: Border.all(color: AppColors.amber.withValues(alpha: .35))),
    child: const Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Icon(Icons.health_and_safety_outlined, color: AppColors.amber),
      SizedBox(width: 10),
      Expanded(child: Text('O resultado orienta a inspeção. Não substitua peças sem a confirmação técnica do equipamento.', style: TextStyle(color: AppColors.text, fontSize: 12, height: 1.35))),
    ]),
  );
}

class StatusBadge extends StatelessWidget {
  const StatusBadge({super.key, required this.level});
  final String level;
  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
    decoration: BoxDecoration(color: AppColors.cyan.withValues(alpha: .14), borderRadius: BorderRadius.circular(99)),
    child: Text(level.toUpperCase(), style: const TextStyle(color: AppColors.cyan, fontSize: 11, fontWeight: FontWeight.w800, letterSpacing: .8)),
  );
}

class ConfidenceGauge extends StatelessWidget {
  const ConfidenceGauge({super.key, required this.value});
  final double value;
  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(18),
    decoration: BoxDecoration(color: AppColors.panel, borderRadius: BorderRadius.circular(18), border: Border.all(color: AppColors.line)),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        const Text('Confiança interna do modelo', style: TextStyle(color: AppColors.muted)),
        Text('${(value * 100).round()}%', style: const TextStyle(color: AppColors.cyan, fontSize: 24, fontWeight: FontWeight.w800)),
      ]),
      const SizedBox(height: 12),
      ClipRRect(borderRadius: BorderRadius.circular(99), child: LinearProgressIndicator(value: value, minHeight: 9, color: AppColors.cyan, backgroundColor: AppColors.line)),
    ]),
  );
}

class ReportSection extends StatelessWidget {
  const ReportSection({super.key, required this.title, required this.icon, required this.items});
  final String title;
  final IconData icon;
  final List<String> items;
  @override
  Widget build(BuildContext context) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(color: AppColors.panel, borderRadius: BorderRadius.circular(18), border: Border.all(color: AppColors.line)),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [Icon(icon, size: 19, color: AppColors.cyan), const SizedBox(width: 9), Text(title, style: const TextStyle(fontWeight: FontWeight.w700))]),
      const SizedBox(height: 12),
      ...items.map((item) => Padding(padding: const EdgeInsets.only(bottom: 8), child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('•  ', style: TextStyle(color: AppColors.cyan)), Expanded(child: Text(item, style: const TextStyle(color: AppColors.muted, height: 1.35)))]))),
    ]),
  );
}

class DiagnosisReport {
  const DiagnosisReport({required this.machineModel, required this.condition, required this.mediaName, required this.mainHypothesis, required this.confidence, required this.confidenceLevel, required this.guidance, required this.priorityParts, required this.possibleCauses, required this.nextSteps});
  final String machineModel;
  final String condition;
  final String mediaName;
  final String mainHypothesis;
  final double confidence;
  final String confidenceLevel;
  final String guidance;
  final List<String> priorityParts;
  final List<String> possibleCauses;
  final List<String> nextSteps;

  String get asText => '''RELATÓRIO DE DIAGNÓSTICO ACÚSTICO

Modelo: $machineModel
Condição: $condition
Arquivo: $mediaName
Hipótese principal: $mainHypothesis
Confiança interna: ${(confidence * 100).round()}%
Classificação: $confidenceLevel

Peças ou conjuntos prioritários:
${priorityParts.map((item) => '- $item').join('\n')}

Possíveis origens:
${possibleCauses.map((item) => '- $item').join('\n')}

Próximos passos:
${nextSteps.asMap().entries.map((entry) => '${entry.key + 1}. ${entry.value}').join('\n')}

Confirme o resultado com inspeção técnica antes de intervir no equipamento.''';
}

class MockDiagnosisService {
  Future<DiagnosisReport> analyze({required String machineModel, required String mediaName, required String operatorReport, required String condition}) async {
    await Future<void>.delayed(const Duration(seconds: 2));
    final text = operatorReport.toLowerCase();
    final bearing = text.contains('vibra') || text.contains('metal') || text.contains('rolamento');
    return DiagnosisReport(
      machineModel: machineModel,
      condition: condition,
      mediaName: mediaName,
      mainHypothesis: bearing ? 'Desgaste de rolamento' : 'Anomalia no conjunto de transmissão',
      confidence: bearing ? .84 : .68,
      confidenceLevel: bearing ? 'Indicação consistente' : 'Indicação moderada',
      guidance: bearing ? 'A assinatura sonora é compatível com registros de desgaste progressivo. Priorize a inspeção antes de manter a máquina em carga elevada.' : 'A assinatura indica uma alteração, mas há hipóteses próximas. Compare a montagem e os componentes de transmissão antes de definir a intervenção.',
      priorityParts: bearing ? ['Rolamento do motor ou eixo principal', 'Alojamento e retentores do conjunto', 'Lubrificante e vedação adjacente'] : ['Correia ou acoplamento de transmissão', 'Polias e pontos de fixação', 'Conjunto de motor e suportes'],
      possibleCauses: bearing ? ['Lubrificação insuficiente ou contaminada', 'Desalinhamento entre eixo e alojamento', 'Vida útil excedida ou carga acima da especificação'] : ['Tensionamento incorreto ou desgaste de correia', 'Folga nos suportes ou desalinhamento', 'Variação de carga durante a operação'],
      nextSteps: const ['Desenergize e siga o procedimento de segurança aplicável.', 'Compare o componente indicado com os sintomas reportados.', 'Registre a confirmação no sistema administrativo para melhorar a base futura.'],
    );
  }
}
