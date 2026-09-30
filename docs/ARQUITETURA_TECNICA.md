# Documentação técnica

## 1. Visão geral

O projeto é uma aplicação web local, de processo único, implementada em Python com Streamlit. O ponto de entrada é [`app.py`](../app.py); nele estão a interface, o pipeline de processamento, a persistência e o treinamento. A aplicação não possui API HTTP de negócio nem banco de dados externo.

```text
Arquivo enviado + laudo
        │
        ├─ ffmpeg → WAV mono 44,1 kHz
        ├─ validação do sinal
        ├─ Silero VAD → detecção de fala
        ├─ Demucs (condicional) → redução de voz
        ├─ 56 atributos acústicos
        └─ spaCy → vetor fixo do laudo
                 │
          concatenação de atributos
                 │
          LightGBM / predict_proba
                 │
       diagnóstico ou registro do caso
```

## 2. Componentes

| Componente | Responsabilidade |
| --- | --- |
| Streamlit | Interface local, estado da sessão, upload, feedback e download de backup. |
| `ffmpeg` | Extração e conversão da primeira trilha de áudio para WAV mono em 44,1 kHz. É encontrado no sistema ou via `imageio-ffmpeg`. |
| Silero VAD | Identifica a proporção e os intervalos de fala humana. |
| Demucs (`htdemucs`) | Remove voz somente nos trechos relevantes detectados pelo VAD. |
| librosa, NumPy e SciPy | Leitura, reamostragem e extração dos atributos acústicos. |
| spaCy (`pt_core_news_sm`) | Produz o vetor fixo do laudo em português. |
| LightGBM + scikit-learn | Classificação, codificação de rótulos e validação cruzada. |
| pandas + joblib | Persistência do dataset CSV e dos artefatos do modelo. |

## 3. Fluxos da aplicação

### 3.1 Alimentação

`alimentar_sistema_em_lote()` valida todos os blocos da interface antes do processamento. Para cada arquivo, o sistema calcula SHA-256, ignora duplicatas, processa o áudio, cria o vetor multimodal e persiste o caso. Ao término do lote, tenta treinar uma única vez; assim, evita retreinos repetidos para cada arquivo.

Um arquivo inválido não encerra o lote: o resultado de cada item é marcado como `ok`, `duplicado` ou `erro`.

### 3.2 Diagnóstico

`gerar_diagnostico()` só é habilitado com um modelo salvo. O arquivo desconhecido percorre o mesmo pipeline de extração, e o vetor resultante é enviado a `predict_proba()`. A confirmação do operador chama `absorver_caso()`, que grava o novo exemplo e retreina se a base estiver apta.

### 3.3 Exclusão e backup

`excluir_casos()` remove a linha do CSV e os arquivos associados; em seguida, retreina ou apaga os artefatos se não houver base suficiente. `gerar_backup_zip()` reúne CSV, áudios, textos e modelos. Arquivos WAV são mantidos sem compressão no ZIP para reduzir tempo de geração.

## 4. Pipeline de áudio

1. `extrair_audio_de_video()` executa `ffmpeg` em arquivo temporário.
2. `obter_metadados_audio()` e `validar_sinal_audio()` verificam duração, canais, silêncio e nível de áudio.
3. `detectar_fala()` calcula a fração de fala. Com menos de 3% (`MIN_SPEECH_RATIO`), o Demucs é ignorado.
4. Havendo fala relevante, `remover_voz_com_demucs()` aplica o modelo apenas aos segmentos necessários, com margens e transição suave.
5. `extrair_features_audio()` avalia janelas de 2 s, avançando 1 s, e usa a média dos vetores por janela. Em áudios mais curtos, usa a própria gravação como janela.

Falhas no VAD ou no Demucs degradam de forma segura: o áudio original continua no fluxo e a interface recebe um aviso.

### 4.1 Atributos acústicos (56)

- 9 atributos de tempo/espectro: RMS, fator de crista, taxa de cruzamento por zero, curtose, assimetria, centróide, largura de banda, roll-off e planura espectral;
- 40 atributos MFCC: média e desvio-padrão dos 20 coeficientes;
- 5 frequências de pico da FFT;
- frequência e força do envelope de amplitude por transformada de Hilbert.

Valores `NaN` e infinitos são convertidos para zero antes de persistir ou inferir.

## 5. Representação textual e fusão multimodal

O laudo passa por normalização de espaços e é convertido com `doc.vector` do spaCy. O vetor tem dimensão fixa `D`, obtida em tempo de execução. O vetor final é:

```text
[56 atributos de áudio | D dimensões do laudo]
```

Essa fusão em nível de atributos mantém uma dimensão estável, ao contrário de um vocabulário TF-IDF que pode crescer. `validar_dimensao_vetor()` impede gravações incompatíveis com o esquema ativo.

## 6. Treinamento e avaliação

- Cada classe só participa do treino ao reunir 10 ou mais exemplos (`MIN_SAMPLES_PER_CLASS`).
- O primeiro modelo exige pelo menos duas classes aptas (`MIN_CLASSES_TREINO`).
- Os rótulos são normalizados sem distinguir caixa, acentos ou espaços excedentes.
- `LabelEncoder` converte rótulos em classes numéricas.
- `LGBMClassifier` é treinado com parâmetros fixos e `random_state=42`; não há busca de hiperparâmetros na interface.
- A acurácia exibida é estimada por validação cruzada estratificada com até cinco dobras. O número de dobras é limitado pelo tamanho da menor classe.

A probabilidade retornada é relativa às classes conhecidas pelo modelo; não comprova a existência de uma falha e não detecta automaticamente classes inéditas.

## 7. Persistência

| Caminho | Conteúdo |
| --- | --- |
| `dataset_master.csv` | Metadados do caso, hash SHA-256, origem, caminhos relativos e vetor multimodal. |
| `audios_conhecidos/` | WAV PCM 16 bits processado, no padrão `ROTULO_ID.wav`. |
| `textos_conhecidos/` | Laudo normalizado, no padrão `ROTULO_ID.txt`. |
| `modelos/modelo_lgbm.joblib` | Classificador treinado. |
| `modelos/label_encoder.joblib` | Mapeamento entre classes e rótulos. |
| `modelos/model_metadata.json` | Colunas, dimensão textual, contagens, classes, assinatura, data e acurácia do treino. |

Os caminhos do dataset são relativos à raiz do projeto para permitir migração. Escritas do CSV, JSON e `joblib` são atômicas: um arquivo temporário é gravado e substitui o destino apenas ao final. Se a escrita do CSV falhar, arquivos recém-criados são removidos para evitar órfãos.

## 8. Concorrência e cache

O processamento e a modificação da base são protegidos por `RLock`, evitando execução concorrente de operações pesadas ou inconsistentes dentro do processo. Recursos imutáveis e o modelo persistido usam `st.cache_resource`; leituras do dataset usam `st.cache_data` com assinatura baseada em data de modificação e tamanho.

Recomenda-se uma instância da aplicação por pasta de dados. Não execute duas cópias de `iniciar.bat` apontando para a mesma instalação.

## 9. Configuração e operação

`.streamlit/config.toml` define upload de até 10 GB, `localhost`, tema claro e telemetria desativada. Constantes relevantes do pipeline ficam no início de `app.py`, incluindo duração de janelas, mínimo de amostras por classe, limiar de fala e tempo limite do ffmpeg.

O `iniciar.bat` para Windows:

1. valida a presença dos arquivos e o tamanho do caminho;
2. encontra ou instala Python x64 compatível;
3. cria `.venv`;
4. instala dependências quando `requirements.txt` é alterado;
5. instala `pt_core_news_sm`;
6. inicia `streamlit run app.py`.

## 10. Segurança, privacidade e manutenção

- O servidor escuta somente em `localhost`; não exponha em rede sem autenticação e revisão de segurança.
- A aba administrativa não possui controle de acesso próprio. A proteção depende da conta do sistema operacional e das permissões da pasta.
- Dados de diagnóstico podem ser sensíveis. Faça backups periódicos e proteja os arquivos ZIP gerados.
- O Demucs pode baixar o modelo na primeira análise com fala. Após isso, é possível operar offline; para impedir consultas externas, use `HF_HUB_OFFLINE=1` depois de preencher o cache.
- Antes de atualizar `app.py` ou dependências, faça backup dos quatro itens de dados. Não copie `.venv` entre computadores.

## 11. Verificação mínima para desenvolvimento

```bash
python -m compileall -q app.py
python -m streamlit run app.py --server.headless true
```

Confirme também o endpoint `/_stcore/health` e a seção **Verificação do ambiente** na barra lateral. Testes com arquivos reais devem cobrir áudio silencioso, arquivo sem trilha de áudio, duplicata, fala humana, classes insuficientes e exclusão de casos.
