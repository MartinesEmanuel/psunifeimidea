# Diagnóstico Acústico — APK de avaliação 1.1.0

Arquivo: `DiagnosticoAcustico-redesign-avaliacao-v1.1.0.apk`. Android 7.0/API 24 ou superior. APK universal para ARMv7, ARM64 e x86_64.

## Instalação

1. Copie o APK para um aparelho de teste e abra-o. Autorize a instalação dessa origem quando o Android solicitar.
2. A instalação usa o nome **Diagnóstico Acústico** e o ícone de onda acústica com barra central carmesim. O identificador é `br.unifei.midea.diagnostico_acustico.redesign`.
3. A versão original permanece instalada e seus dados não são migrados nem compartilhados. Confira o identificador nas informações do aplicativo caso apareçam dois nomes iguais.
4. Leia o guia, preencha modelo/laudo e grave ou importe o áudio. Conceda acesso ao microfone quando solicitado para gravação.
5. Para receber a IA e sincronizar, use o código de conexão e uma conta de teste fornecidos pela Central. Sem pacote de IA, o aplicativo informa a ausência do modelo e permite salvar o caso offline; não simula uma classificação.

## O que está incluído

Oito telas redesenhadas, tipografia e componentes consistentes, logos institucionais preservados, layouts roláveis/adaptativos, contraste e alvos de toque aprimorados, movimento reduzido e novo acabamento vetorial do ícone Android.

Os fontes compiláveis estão em `../mobile_app/`; a auditoria e o relatório com capturas, provas e limitações estão em `../docs/REDESIGN_MOBILE.md`.

## Natureza da entrega

Esta é uma **reconstrução para avaliação**, baseada no APK e no pseudocódigo recuperado, assinada com chave local de desenvolvimento. Não é uma atualização oficial assinada nem o código-fonte original. Não há comprovação de equivalência integral ao binário anterior.

O kit numérico, persistência, formatos de envio e receptor Python foram exercitados. Reamostragem em outras taxas, alternativas de captura, codecs de diferentes fabricantes, conta real da Central, microfone físico, TalkBack e desempenho em campo exigem validação adicional. Avalie em aparelho e conta de teste antes do uso operacional.

O arquivo original `DiagnosticoAcustico-app-v1.0.0.apk` permanece intacto. A soma SHA-256 do novo artefato acompanha a entrega para conferência de integridade.
