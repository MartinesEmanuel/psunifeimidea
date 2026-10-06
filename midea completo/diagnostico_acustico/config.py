"""Configuração global: pastas, constantes e parâmetros."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("diagnostico_acustico")

# ==============================================================================
# 1. CONFIGURAÇÃO GLOBAL / CONSTANTES
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # pasta do app.py
AUDIOS_DIR = BASE_DIR / "audios_conhecidos"
TEXTOS_DIR = BASE_DIR / "textos_conhecidos"
MODELOS_DIR = BASE_DIR / "modelos"

DATASET_PATH = BASE_DIR / "dataset_master.csv"
MODEL_PATH = MODELOS_DIR / "modelo_lgbm.joblib"
ENCODER_PATH = MODELOS_DIR / "label_encoder.joblib"
METADATA_PATH = MODELOS_DIR / "model_metadata.json"

for _dir in (AUDIOS_DIR, TEXTOS_DIR, MODELOS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Textos de cada defeito (causas, o que verificar, o que fazer), escritos pelo
# administrador e enviados aos celulares junto com a IA.
BASE_CONHECIMENTO_PATH = BASE_DIR / "base_conhecimento.json"
# Casos gravados nos celulares, recebidos pela caixa postal e aguardando decisão.
CAIXA_ENTRADA_DIR = BASE_DIR / "caixa_de_entrada"
# Pacotes (IA + textos + peças) gerados para os celulares.
PUBLICACOES_DIR = BASE_DIR / "publicacoes"
# Endereço e chaves da caixa postal na internet. Fica fora do backup: a chave
# de serviço dá acesso total à caixa postal e não deve sair deste computador.
SINCRONIZACAO_PATH = BASE_DIR / ".streamlit" / "sincronizacao.json"
BUCKET_CAIXA_POSTAL = "caixa-postal"
FORMATO_PACOTE = 1        # versão do formato do pacote enviado aos celulares
FORMATO_CASO_CELULAR = 1  # versão do formato do caso enviado pelos celulares

SPACY_MODEL_NAME = "pt_core_news_sm"
MIN_SAMPLES_PER_CLASS = 10  # Regra de acumulação (Seção 3 do briefing)
MIN_CLASSES_TREINO = 2      # um classificador precisa de pelo menos 2 classes

WINDOW_SEC = 2.0
HOP_SEC = 1.0
N_MFCC = 20
N_PEAKS = 5

VAD_SAMPLE_RATE = 16000
MIN_SPEECH_RATIO = 0.03  # >=3% de fala detectada no áudio => aciona o Demucs

DEMUCS_MODEL_NAME = "htdemucs"
DEMUCS_MARGEM_S = 1.0          # contexto extra em volta de cada trecho de fala enviado ao Demucs
DEMUCS_UNIR_TRECHOS_S = 2.0    # trechos de fala mais próximos que isso são processados juntos
DEMUCS_COBERTURA_AUDIO_TODO = 0.6  # se a fala (com margens) cobre >= 60% do áudio, processa tudo
CROSSFADE_S = 0.05             # transição suave entre trechos originais e trechos sem voz

FFMPEG_TARGET_SR = 44100
FFMPEG_TIMEOUT_S = 600

MIN_AUDIO_DURATION_S = 0.5
PICO_AUDIO_MUDO = 1e-4   # abaixo disso a trilha de áudio é considerada muda
RMS_AUDIO_BAIXO = 1e-3   # abaixo disso o volume é considerado baixo (apenas aviso)

MAX_VIDEO_DURATION_WARNING_S = 180  # 3 minutos -- apenas recomendação, não é um bloqueio

TIPOS_VIDEO = ["mp4", "mov", "m4v", "avi", "mkv", "webm", "3gp", "wmv"]
TIPOS_AUDIO = ["wav", "mp3", "m4a", "aac", "ogg", "flac"]
TIPOS_ACEITOS = TIPOS_VIDEO + TIPOS_AUDIO

N_FOLDS_MAX_CV = 5  # validação cruzada usada para estimar a acurácia após cada treino

# Categoria dos vídeos da máquina funcionando normal. Ela é treinada como as
# outras, e é o que permite à IA responder "não há defeito".
ROTULO_REFERENCIA = "Referência"
# Nomes que, digitados como problema, são entendidos como a Referência
# (comparação sem maiúsculas, sem acentos e sem espaços extras).
APELIDOS_REFERENCIA = (
    "referencia", "normal", "funcionamento normal", "maquina normal", "maquina funcionando normal",
    "sem defeito", "sem problema", "ok", "saudavel",
)

# Senha do modo administrador: guardada cifrada (PBKDF2-SHA256), fora do código
# e fora do backup -- cada computador tem a sua.
SENHA_ADMIN_PATH = BASE_DIR / ".streamlit" / "senha_admin.json"
SENHA_ADMIN_ITERACOES = 600_000
SENHA_ADMIN_MIN_CARACTERES = 6

# Hiperparâmetros FIXOS do LightGBM (Otimização 2: nada de GridSearchCV na interface).
# num_leaves/min_child_samples reduzidos em relação ao default porque o regime de
# dados aqui é pequeno por natureza (classes entram no treino a partir de
# MIN_SAMPLES_PER_CLASS = 10 amostras).
LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 5,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

# Callback de progresso usado pelo pipeline: (texto da etapa, fração 0..1 do item atual).
Etapa = Callable[[str, float], None]


def _sem_progresso(_texto: str, _fracao: float) -> None:
    """Callback de progresso vazio (uso fora da interface, ex.: testes)."""
