"""
================================================================================
 Sistema de Diagnóstico Acústico de Falhas Mecânicas — app.py
================================================================================
Aplicação Streamlit multimodal (áudio + texto) para diagnosticar falhas
mecânicas em equipamentos a partir de vídeos e laudos textuais.

COMO RODAR (Windows):
    Clique duas vezes em iniciar.bat -- ele instala o que faltar e abre o app.

COMO RODAR (manual / Linux / macOS):
    1. Instale o ffmpeg no sistema operacional (obrigatório, fora do pip):
         - Ubuntu/Debian: sudo apt install ffmpeg
         - macOS (brew):  brew install ffmpeg
         - Windows:       winget install ffmpeg -e --id Gyan.FFmpeg
    2. Crie um ambiente virtual e instale as dependências (ver requirements.txt):
         pip install -r requirements.txt
    3. Baixe o modelo de linguagem em português do spaCy:
         python -m spacy download pt_core_news_sm
    4. Mantenha a pasta ".streamlit/config.toml" ao lado deste app.py -- ela
       remove o limite padrão de 200 MB por upload e define o tema visual.
    5. Rode a aplicação:
         streamlit run app.py

    Na primeira vez que um vídeo com fala humana for processado, o modelo do
    Demucs (~80 MB) é baixado da internet e fica em cache no computador.

Estrutura criada automaticamente ao lado deste arquivo:
    audios_conhecidos/   -> áudios rotulados,  nomeados [ROTULO]_[UUID].wav
    textos_conhecidos/   -> laudos rotulados,  nomeados [ROTULO]_[UUID].txt
    modelos/             -> modelo_lgbm.joblib, label_encoder.joblib, metadata
    dataset_master.csv   -> tabela mestre (UUID, rótulo, caminhos, metadados, features)

MIGRAR PARA OUTRO COMPUTADOR (sem perder o treino):
    Copie estes 4 itens para a mesma pasta do app.py na outra máquina:
        dataset_master.csv, audios_conhecidos/, textos_conhecidos/, modelos/
    Ou use o botão "Baixar backup (.zip)" na barra lateral do próprio app,
    que empacota os 4 itens automaticamente. Todos os caminhos salvos no CSV
    são relativos à pasta do app.py, então a cópia funciona em qualquer PC.
================================================================================
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import joblib
import librosa  # usado SOMENTE para algoritmos de extração (mfcc, espectrais) e reamostragem
import numpy as np
import pandas as pd
import soundfile as sf
import spacy
import streamlit as st
import torch
from lightgbm import LGBMClassifier
from scipy.signal import find_peaks, hilbert
from scipy.stats import kurtosis, skew
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("diagnostico_acustico")

# Import "leve" e opcional do Silero VAD: se o pacote (ou o torchaudio, que ele
# importa) não estiver disponível, o app continua de pé e os vídeos apenas
# deixam de passar pela remoção de voz -- o usuário é avisado no resultado.
try:
    import silero_vad as _silero_pkg
    from silero_vad import get_speech_timestamps as _silero_get_speech_timestamps

    _SILERO_DISPONIVEL = True
    _SILERO_ERRO_IMPORT = ""
except Exception as _exc_silero:  # pragma: no cover - depende do ambiente de execução
    _SILERO_DISPONIVEL = False
    _SILERO_ERRO_IMPORT = str(_exc_silero)


# ==============================================================================
# 1. CONFIGURAÇÃO GLOBAL / CONSTANTES
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
AUDIOS_DIR = BASE_DIR / "audios_conhecidos"
TEXTOS_DIR = BASE_DIR / "textos_conhecidos"
MODELOS_DIR = BASE_DIR / "modelos"

DATASET_PATH = BASE_DIR / "dataset_master.csv"
MODEL_PATH = MODELOS_DIR / "modelo_lgbm.joblib"
ENCODER_PATH = MODELOS_DIR / "label_encoder.joblib"
METADATA_PATH = MODELOS_DIR / "model_metadata.json"

for _dir in (AUDIOS_DIR, TEXTOS_DIR, MODELOS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

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


# ==============================================================================
# 2. EXCEÇÕES DO DOMÍNIO
# ==============================================================================

class PipelineError(Exception):
    """Erro genérico e tratável do pipeline de diagnóstico acústico."""


class ExtracaoAudioError(PipelineError):
    """Falha ao extrair ou ler o áudio de um vídeo."""


class ExtracaoFeaturesError(PipelineError):
    """Falha ao calcular features acústicas, textuais ou o vetor multimodal."""


class ModeloIndisponivelError(PipelineError):
    """Um recurso de modelo (spaCy, Silero VAD, Demucs, LightGBM) não está disponível."""


class DatasetError(PipelineError):
    """Falha de leitura, escrita ou consistência do dataset_master.csv."""


# ==============================================================================
# 3. RECURSOS CACHEADOS (spaCy / Silero VAD / Demucs) E CONTROLE DE CONCORRÊNCIA
# ==============================================================================

@st.cache_resource(show_spinner=False)
def _trava_processamento() -> threading.RLock:
    """Trava única, compartilhada por todas as abas/sessões abertas do app.

    O Streamlit reexecuta o script a cada interação, então uma trava criada no
    nível do módulo seria recriada toda vez; via cache_resource ela é a mesma
    para sempre. Ela serializa o processamento pesado (VAD e Demucs guardam
    estado interno) e as escritas no dataset_master.csv / pasta modelos/."""
    return threading.RLock()


@contextlib.contextmanager
def _threads_torch(n: int) -> Iterator[None]:
    """Ajusta temporariamente o número de threads do PyTorch. O Silero VAD é
    mais rápido com 1 thread; o Demucs precisa de todos os núcleos."""
    anterior = torch.get_num_threads()
    torch.set_num_threads(max(1, int(n)))
    try:
        yield
    finally:
        torch.set_num_threads(anterior)


@st.cache_resource(show_spinner="Carregando modelo de linguagem (spaCy pt_core_news_sm)...")
def carregar_spacy() -> "spacy.language.Language":
    """Carrega o pipeline spaCy em português usado para gerar `doc.vector`."""
    try:
        return spacy.load(SPACY_MODEL_NAME)
    except OSError as exc:
        raise ModeloIndisponivelError(
            f"Modelo spaCy '{SPACY_MODEL_NAME}' não está instalado. "
            f"Rode: python -m spacy download {SPACY_MODEL_NAME}"
        ) from exc


@st.cache_data(show_spinner=False)
def obter_dimensao_vetor_texto() -> int:
    """Descobre, uma única vez, o tamanho fixo do vetor textual do spaCy carregado."""
    nlp = carregar_spacy()
    return len(nlp("teste de dimensionalidade").vector)


@st.cache_resource(show_spinner="Carregando modelo Silero VAD...")
def carregar_vad() -> Any:
    """Carrega o modelo Silero VAD (detecção de voz) uma única vez por processo.

    IMPORTANTE: não usamos `silero_vad.load_silero_vad()` porque ele chama
    `torch.jit.load(caminho)`, cujo código C++ abre o arquivo com a codificação
    ANSI do Windows e FALHA quando o caminho tem acentos (ex.: a pasta "Área de
    Trabalho"). Lemos os bytes pelo Python (que lida bem com Unicode) e
    entregamos um buffer em memória ao PyTorch."""
    if not _SILERO_DISPONIVEL:
        raise ModeloIndisponivelError(
            f"Pacote 'silero-vad' indisponível ({_SILERO_ERRO_IMPORT}). Rode: pip install silero-vad"
        )
    caminho_modelo = Path(_silero_pkg.__file__).resolve().parent / "data" / "silero_vad.jit"
    try:
        with _threads_torch(1):
            modelo = torch.jit.load(io.BytesIO(caminho_modelo.read_bytes()), map_location="cpu")
        modelo.eval()
    except Exception as exc:
        raise ModeloIndisponivelError(f"Falha ao carregar o Silero VAD: {exc}") from exc
    return modelo


@st.cache_resource(
    show_spinner="Carregando o Demucs (na primeira vez o modelo, ~80 MB, é baixado da internet)..."
)
def carregar_demucs() -> Tuple[Any, Callable[..., Any]]:
    """Carrega o modelo de separação de fontes do Demucs uma única vez por processo.

    Roda dentro do próprio Python do app (e não via linha de comando `demucs`),
    o que garante que seja usado o Demucs do ambiente virtual e que a leitura e
    a escrita de áudio fiquem a cargo do soundfile -- sem depender do torchaudio."""
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as exc:
        raise ModeloIndisponivelError(f"Pacote 'demucs' indisponível: {exc}") from exc
    try:
        modelo = get_model(DEMUCS_MODEL_NAME)
        modelo.eval()
    except Exception as exc:
        raise ModeloIndisponivelError(
            f"não foi possível carregar o modelo '{DEMUCS_MODEL_NAME}' do Demucs -- na primeira "
            f"execução ele precisa ser baixado da internet ({exc})"
        ) from exc
    return modelo, apply_model


# ==============================================================================
# 4. E/S DE ÁUDIO (extração de vídeo, metadados, leitura de amostras)
# ==============================================================================

@dataclass(frozen=True)
class AudioMetadata:
    duracao_s: float
    sample_rate: int
    canais: int


def localizar_ffmpeg() -> Optional[str]:
    """Procura o ffmpeg no PATH e, no Windows, também nas pastas do winget --
    útil quando o ffmpeg acabou de ser instalado e o PATH desta janela ainda
    não foi atualizado. Se não houver ffmpeg no sistema, usa o que vem dentro
    do pacote pip `imageio-ffmpeg` (ver requirements.txt), o que permite rodar
    o app em qualquer computador sem instalar o ffmpeg à parte."""
    caminho = shutil.which("ffmpeg")
    if caminho:
        return caminho
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidatos = [local / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"]
        candidatos += sorted((local / "Microsoft" / "WinGet" / "Packages").glob("Gyan.FFmpeg*/*/bin/ffmpeg.exe"))
        for candidato in candidatos:
            if candidato.is_file():
                return str(candidato)
    try:
        import imageio_ffmpeg

        embutido = imageio_ffmpeg.get_ffmpeg_exe()
        if embutido and Path(embutido).is_file():
            return embutido
    except Exception:  # pacote ausente ou binário indisponível para esta plataforma
        pass
    return None


def _rodar_subprocesso(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, **kwargs,
    )


def _traduzir_erro_ffmpeg(stderr: Optional[str]) -> str:
    """Converte as mensagens mais comuns do ffmpeg em algo que o usuário entenda."""
    texto = (stderr or "").strip()
    baixo = texto.lower()
    if "matches no streams" in baixo or "does not contain any stream" in baixo:
        return "o arquivo não tem trilha de áudio (foi gravado sem som). Grave novamente com o microfone ativo."
    if "invalid data found" in baixo or "moov atom not found" in baixo:
        return "o arquivo parece corrompido, incompleto ou está num formato que o ffmpeg não reconhece."
    return f"falha ao extrair o áudio com o ffmpeg: {texto[-400:] or 'erro desconhecido'}"


def extrair_audio_de_video(video_bytes: bytes, sufixo_video: str) -> Path:
    """Salva o vídeo em um arquivo temporário e extrai a primeira trilha de
    áudio como .wav mono 44,1 kHz via ffmpeg. Quem chama deve apagar o .wav."""
    ffmpeg = localizar_ffmpeg()
    if ffmpeg is None:
        raise ExtracaoAudioError(
            "ffmpeg não encontrado. Rode o iniciar.bat de novo ou instale com: "
            "winget install ffmpeg -e --id Gyan.FFmpeg"
        )
    if not video_bytes:
        raise ExtracaoAudioError("o arquivo enviado está vazio (0 bytes).")

    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=sufixo_video, prefix="diag_video_")
    tmp_wav_path = Path(tempfile.gettempdir()) / f"diag_audio_{uuid.uuid4().hex}.wav"
    try:
        tmp_video.write(video_bytes)
        tmp_video.close()
        cmd = [
            ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-i", tmp_video.name,
            "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(FFMPEG_TARGET_SR), "-c:a", "pcm_s16le",
            str(tmp_wav_path),
        ]
        try:
            resultado = _rodar_subprocesso(cmd, FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            tmp_wav_path.unlink(missing_ok=True)
            raise ExtracaoAudioError("tempo limite excedido ao extrair o áudio com o ffmpeg.") from exc

        if resultado.returncode != 0 or not tmp_wav_path.exists() or tmp_wav_path.stat().st_size == 0:
            tmp_wav_path.unlink(missing_ok=True)
            raise ExtracaoAudioError(_traduzir_erro_ffmpeg(resultado.stderr))
        return tmp_wav_path
    finally:
        tmp_video.close()
        Path(tmp_video.name).unlink(missing_ok=True)


def obter_metadados_audio(caminho_wav: Path) -> AudioMetadata:
    """OTIMIZAÇÃO 1: usa EXCLUSIVAMENTE soundfile.info() (carregamento instantâneo,
    sem decodificar nenhuma amostra) para descobrir duração, sample_rate e canais."""
    try:
        info = sf.info(str(caminho_wav))
    except Exception as exc:
        raise ExtracaoAudioError(f"não foi possível ler metadados de '{caminho_wav.name}': {exc}") from exc
    duracao = info.frames / info.samplerate if info.samplerate else 0.0
    return AudioMetadata(duracao_s=duracao, sample_rate=info.samplerate, canais=info.channels)


def carregar_amostras_audio(caminho_wav: Path, sr_alvo: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Lê as amostras de áudio via soundfile (nunca librosa.load), converte para
    mono float32 e, se solicitado, reamostra para `sr_alvo` usando librosa.resample."""
    try:
        dados, sr = sf.read(str(caminho_wav), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ExtracaoAudioError(f"não foi possível ler as amostras de '{caminho_wav.name}': {exc}") from exc

    if dados.ndim > 1:
        dados = dados.mean(axis=1).astype(np.float32)

    if sr_alvo is not None and sr != sr_alvo:
        dados = librosa.resample(dados, orig_sr=sr, target_sr=sr_alvo).astype(np.float32)
        sr = sr_alvo

    return dados, sr


def validar_sinal_audio(y: np.ndarray, sr: int) -> List[str]:
    """Barra trilhas mudas ou curtas demais (que só gerariam features lixo) e
    devolve avisos não bloqueantes, como volume muito baixo."""
    duracao = len(y) / sr if sr else 0.0
    if duracao < MIN_AUDIO_DURATION_S:
        raise ExtracaoAudioError(
            f"o áudio tem só {duracao:.2f} s -- é preciso pelo menos {MIN_AUDIO_DURATION_S:.1f} s."
        )
    pico = float(np.max(np.abs(y))) if len(y) else 0.0
    if pico < PICO_AUDIO_MUDO:
        raise ExtracaoAudioError("a trilha de áudio está muda (silêncio absoluto). Verifique o microfone.")
    avisos: List[str] = []
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    if rms < RMS_AUDIO_BAIXO:
        avisos.append("volume muito baixo -- se possível, grave mais perto do equipamento.")
    return avisos


# ==============================================================================
# 4b. PRÉ-PROCESSAMENTO: DETECÇÃO (Silero VAD) E REMOÇÃO (Demucs) DE VOZ HUMANA
# ==============================================================================

def detectar_fala(y: np.ndarray, sr: int) -> Tuple[float, List[Tuple[float, float]]]:
    """OTIMIZAÇÃO 3: roda o Silero VAD apenas sobre o único áudio recém-extraído
    (nunca em loop varrendo pastas). Devolve a fração do áudio com fala e os
    trechos de fala, em segundos."""
    modelo_vad = carregar_vad()
    y16 = y if sr == VAD_SAMPLE_RATE else librosa.resample(y, orig_sr=sr, target_sr=VAD_SAMPLE_RATE)
    tensor_audio = torch.from_numpy(np.ascontiguousarray(y16, dtype=np.float32))
    with _threads_torch(1), torch.no_grad():
        timestamps = _silero_get_speech_timestamps(
            tensor_audio, modelo_vad, sampling_rate=VAD_SAMPLE_RATE, return_seconds=True
        )
    duracao = len(y16) / VAD_SAMPLE_RATE
    trechos = [(float(t["start"]), float(t["end"])) for t in timestamps]
    duracao_fala = sum(fim - inicio for inicio, fim in trechos)
    return (duracao_fala / duracao if duracao > 0 else 0.0), trechos


def _segmentos_para_demucs(trechos_fala: List[Tuple[float, float]], duracao: float) -> List[Tuple[float, float]]:
    """Agrupa os trechos de fala (com margem de contexto) nos intervalos que o
    Demucs deve processar. O resto do áudio fica intocado: além de mais rápido,
    evita artefatos de separação onde não havia voz."""
    segmentos: List[List[float]] = []
    for inicio, fim in sorted(trechos_fala):
        inicio, fim = max(0.0, inicio - DEMUCS_MARGEM_S), min(duracao, fim + DEMUCS_MARGEM_S)
        if segmentos and inicio - segmentos[-1][1] <= DEMUCS_UNIR_TRECHOS_S:
            segmentos[-1][1] = max(segmentos[-1][1], fim)
        else:
            segmentos.append([inicio, fim])
    cobertura = sum(fim - inicio for inicio, fim in segmentos)
    if duracao > 0 and cobertura / duracao >= DEMUCS_COBERTURA_AUDIO_TODO:
        return [(0.0, duracao)]
    return [(inicio, fim) for inicio, fim in segmentos]


def _separar_sem_voz(y: np.ndarray, sr: int, modelo: Any, apply_model: Callable[..., Any]) -> np.ndarray:
    """Equivalente a `demucs --two-stems vocals`: devolve a soma de todas as
    fontes exceto "vocals", em mono e no mesmo sample rate/tamanho da entrada."""
    sr_modelo = int(modelo.samplerate)
    x = y if sr == sr_modelo else librosa.resample(y, orig_sr=sr, target_sr=sr_modelo)
    wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    wav = wav.unsqueeze(0).expand(int(modelo.audio_channels), -1).contiguous()

    referencia = wav.mean(0)
    media, desvio = referencia.mean(), referencia.std()
    if float(desvio) < 1e-8:  # trecho silencioso: nada a separar
        return y.copy()

    with _threads_torch(os.cpu_count() or 1), torch.no_grad():
        fontes = apply_model(
            modelo, ((wav - media) / desvio)[None], device="cpu",
            shifts=0, split=True, overlap=0.25, progress=False,
        )[0]
    fontes = fontes * desvio + media

    indice_voz = list(modelo.sources).index("vocals")
    sem_voz = sum(fontes[i] for i in range(fontes.shape[0]) if i != indice_voz)
    sem_voz = sem_voz.mean(0).cpu().numpy().astype(np.float32)
    if sr != sr_modelo:
        sem_voz = librosa.resample(sem_voz, orig_sr=sr_modelo, target_sr=sr).astype(np.float32)

    if len(sem_voz) < len(y):
        sem_voz = np.pad(sem_voz, (0, len(y) - len(sem_voz)))
    return sem_voz[: len(y)]


def remover_voz_com_demucs(y: np.ndarray, sr: int, trechos_fala: List[Tuple[float, float]]) -> np.ndarray:
    """OTIMIZAÇÃO 3: separação de fontes pontual (Demucs) rodando só sobre este
    áudio -- e, dentro dele, só sobre os trechos em que o VAD encontrou fala."""
    modelo, apply_model = carregar_demucs()
    duracao = len(y) / sr
    saida = y.copy()
    n_fade = int(CROSSFADE_S * sr)

    for inicio_s, fim_s in _segmentos_para_demucs(trechos_fala, duracao):
        a, b = int(inicio_s * sr), min(len(y), int(round(fim_s * sr)))
        if b - a < int(0.1 * sr):
            continue
        trecho_sem_voz = _separar_sem_voz(y[a:b], sr, modelo, apply_model)
        f = min(n_fade, (b - a) // 2)
        if f > 0:  # transição suave com o áudio original nas bordas internas
            rampa = np.linspace(0.0, 1.0, f, dtype=np.float32)
            if a > 0:
                trecho_sem_voz[:f] = y[a:a + f] * (1 - rampa) + trecho_sem_voz[:f] * rampa
            if b < len(y):
                trecho_sem_voz[-f:] = trecho_sem_voz[-f:] * (1 - rampa) + y[b - f:b] * rampa
        saida[a:b] = trecho_sem_voz
    return saida


@dataclass
class ResultadoPreprocessamento:
    audio: np.ndarray
    ratio_fala: Optional[float]  # None = VAD indisponível
    demucs_aplicado: bool
    avisos: List[str] = field(default_factory=list)


def preprocessar_audio_pipeline(y: np.ndarray, sr: int, etapa: Etapa = _sem_progresso) -> ResultadoPreprocessamento:
    """Aplica VAD e, se necessário, Demucs -- SEMPRE de forma pontual sobre um
    único áudio. Se algum dos dois modelos estiver indisponível, o áudio
    original é usado e o motivo vai para a lista de avisos (degradação suave)."""
    etapa("detectando voz humana (Silero VAD)", 0.15)
    try:
        ratio_fala, trechos = detectar_fala(y, sr)
    except Exception as exc:
        logger.exception("Silero VAD falhou")
        return ResultadoPreprocessamento(
            audio=y, ratio_fala=None, demucs_aplicado=False,
            avisos=[f"a detecção de voz falhou ({exc}); o áudio foi usado sem remoção de voz."],
        )

    if ratio_fala < MIN_SPEECH_RATIO:
        logger.info("Sem voz relevante detectada (%.1f%%) -- Demucs pulado.", ratio_fala * 100)
        return ResultadoPreprocessamento(audio=y, ratio_fala=ratio_fala, demucs_aplicado=False)

    logger.info("Voz detectada (%.1f%% do áudio) -- aplicando Demucs.", ratio_fala * 100)
    etapa(f"removendo a voz humana detectada em {ratio_fala:.0%} do áudio (Demucs)", 0.3)
    try:
        audio_sem_voz = remover_voz_com_demucs(y, sr, trechos)
    except Exception as exc:
        logger.exception("Demucs falhou")
        return ResultadoPreprocessamento(
            audio=y, ratio_fala=ratio_fala, demucs_aplicado=False,
            avisos=[f"havia voz em {ratio_fala:.0%} do áudio, mas o Demucs falhou ({exc}); o áudio original foi usado."],
        )
    return ResultadoPreprocessamento(audio=audio_sem_voz, ratio_fala=ratio_fala, demucs_aplicado=True)


# ==============================================================================
# 5. EXTRAÇÃO DE FEATURES ACÚSTICAS (56 parâmetros)
# ==============================================================================

AUDIO_FEATURE_COLUMNS: List[str] = (
    ["rms", "crest_factor", "zcr", "kurtosis", "skewness",
     "spec_centroid", "spec_bandwidth", "spec_rolloff", "spec_flatness"]
    + [f"mfcc{i + 1}_{stat}" for i in range(N_MFCC) for stat in ("mean", "std")]
    + [f"peak_freq_{i + 1}_hz" for i in range(N_PEAKS)]
    + ["envelope_freq_hz", "envelope_strength"]
)
assert len(AUDIO_FEATURE_COLUMNS) == 56, "A extração deve gerar exatamente 56 parâmetros acústicos."


def _extrair_parametros_janela(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extrai os 56 parâmetros acústicos de uma única janela de áudio."""
    eps = 1e-9

    # --- Domínio do tempo ---
    rms = float(np.sqrt(np.mean(np.square(y))) + eps)
    crest_factor = float(np.max(np.abs(y)) / rms) if rms > 0 else 0.0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    kurt = float(kurtosis(y))
    skw = float(skew(y))

    # --- Domínio da frequência (espectrais) ---
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # --- MFCCs (timbre) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_feats: Dict[str, float] = {}
    for i in range(N_MFCC):
        mfcc_feats[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
        mfcc_feats[f"mfcc{i + 1}_std"] = float(np.std(mfcc[i]))

    # --- Picos de frequência dominantes (top 5) ---
    fft_vals = np.abs(np.fft.rfft(y))
    fft_freqs = np.fft.rfftfreq(len(y), 1 / sr)
    peaks_idx, _ = find_peaks(fft_vals, distance=5)
    if len(peaks_idx) > 0:
        top_peaks = peaks_idx[np.argsort(fft_vals[peaks_idx])[::-1][:N_PEAKS]]
        peak_freqs = sorted(fft_freqs[top_peaks].tolist())
    else:
        peak_freqs = []
    peak_freqs = (peak_freqs + [0.0] * N_PEAKS)[:N_PEAKS]

    # --- Envelope (modulação de amplitude) ---
    envelope = np.abs(hilbert(y))
    env_centered = envelope - np.mean(envelope)
    env_fft = np.abs(np.fft.rfft(env_centered))
    env_freqs = np.fft.rfftfreq(len(env_centered), 1 / sr)
    if len(env_fft) > 1:
        env_fft_no_dc = env_fft[1:]
        env_freqs_no_dc = env_freqs[1:]
        idx_max = int(np.argmax(env_fft_no_dc))
        envelope_freq_hz = float(env_freqs_no_dc[idx_max])
        envelope_strength = float(env_fft_no_dc[idx_max] / (np.sum(env_fft_no_dc) + eps))
    else:
        envelope_freq_hz, envelope_strength = 0.0, 0.0

    return {
        "rms": rms, "crest_factor": crest_factor, "zcr": zcr,
        "kurtosis": kurt, "skewness": skw,
        "spec_centroid": spec_centroid, "spec_bandwidth": spec_bandwidth,
        "spec_rolloff": spec_rolloff, "spec_flatness": spec_flatness,
        **mfcc_feats,
        "peak_freq_1_hz": peak_freqs[0], "peak_freq_2_hz": peak_freqs[1],
        "peak_freq_3_hz": peak_freqs[2], "peak_freq_4_hz": peak_freqs[3],
        "peak_freq_5_hz": peak_freqs[4],
        "envelope_freq_hz": envelope_freq_hz, "envelope_strength": envelope_strength,
    }


def extrair_features_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Extrai o vetor de 56 features acústicas, calculado como a média das
    janelas deslizantes (WINDOW_SEC/HOP_SEC) sobre o áudio inteiro."""
    try:
        duracao = len(y) / sr if sr else 0.0

        janelas: List[Dict[str, float]] = []
        t = 0.0
        while t + WINDOW_SEC <= duracao:
            inicio, fim = int(t * sr), int((t + WINDOW_SEC) * sr)
            janelas.append(_extrair_parametros_janela(y[inicio:fim], sr))
            t += HOP_SEC
        if not janelas:
            janelas = [_extrair_parametros_janela(y, sr)]

        df_janelas = pd.DataFrame(janelas)[AUDIO_FEATURE_COLUMNS]
        vetor_medio = df_janelas.mean(axis=0).to_numpy(dtype=np.float64)
        return np.nan_to_num(vetor_medio, nan=0.0, posinf=0.0, neginf=0.0)
    except PipelineError:
        raise
    except Exception as exc:
        raise ExtracaoFeaturesError(f"falha ao extrair features acústicas: {exc}") from exc


# ==============================================================================
# 6. PROCESSAMENTO DE TEXTO (vetor fixo via spaCy -- SEM TF-IDF)
# ==============================================================================

def limpar_texto(texto: str) -> str:
    """Normalização mínima e segura do laudo textual (sem alterar semântica)."""
    return " ".join((texto or "").strip().split())


def extrair_vetor_texto(texto: str) -> np.ndarray:
    """MULTIMODALIDADE FIXA: usa EXCLUSIVAMENTE `doc.vector` do spaCy
    (pt_core_news_sm). Diferente do TF-IDF -- cujo vocabulário (e, portanto, o
    número de colunas) cresce a cada palavra inédita, gerando shape mismatch em
    aprendizado incremental -- o `doc.vector` tem SEMPRE o mesmo tamanho fixo,
    não importa o texto de entrada."""
    try:
        nlp = carregar_spacy()
        dim = obter_dimensao_vetor_texto()
        texto_limpo = limpar_texto(texto)
        if not texto_limpo:
            return np.zeros(dim, dtype=np.float64)
        doc = nlp(texto_limpo)
        return doc.vector.astype(np.float64)
    except PipelineError:
        raise
    except Exception as exc:
        raise ExtracaoFeaturesError(f"falha ao extrair o vetor textual: {exc}") from exc


# ==============================================================================
# 7. FUSÃO MULTIMODAL (Áudio + Texto)
# ==============================================================================

def nomes_colunas_features(dim_texto: int) -> List[str]:
    """Nomes de coluna fixos para o vetor multimodal: 56 features de áudio +
    `dim_texto` dimensões do vetor textual do spaCy."""
    return list(AUDIO_FEATURE_COLUMNS) + [f"texto_vec_{i:03d}" for i in range(dim_texto)]


def montar_vetor_multimodal(features_audio: np.ndarray, vetor_texto: np.ndarray) -> np.ndarray:
    """
    ==== FUSÃO MULTIMODAL POR CONCATENAÇÃO (Áudio + Texto) ====

    Esta é a etapa central da multimodalidade do sistema (no código original
    chamada de "late fusion"; o termo acadêmico mais preciso é fusão em nível
    de atributos, ou *feature-level fusion*, porque existe um único
    classificador depois da concatenação):

        vetor_final = [ Audio_Features_56 | Text_Vector_D ]

    Por que essa abordagem é a correta para este problema:

      1. Tamanho fixo e determinístico: 56 (áudio) + D (texto, fixo pelo
         modelo spaCy carregado) = 56 + D colunas SEMPRE, não importa o
         conteúdo do texto ou a duração do áudio de entrada. É exatamente
         isso que permite retreinar o LightGBM incrementalmente (a cada novo
         caso absorvido) sem NENHUM risco de "shape mismatch" -- o problema
         clássico de usar TF-IDF em online learning, onde o vocabulário (e,
         portanto, o número de colunas de entrada) cresce a cada palavra
         inédita vista pelo sistema.

      2. Cada modalidade preserva sua própria semântica até chegar ao
         classificador: as 56 primeiras posições carregam a "assinatura
         vibracional/acústica" da falha mecânica (RMS, MFCCs, picos
         espectrais, envelope de amplitude etc.); as posições seguintes
         carregam o "contexto textual" do laudo (sintomas relatados pelo
         operador, palavras-chave do problema). O LightGBM, por ser baseado
         em árvores de decisão, aprende sozinho -- através dos splits em cada
         nó -- QUAIS dimensões (de qualquer uma das duas modalidades) são
         mais discriminativas para cada classe de defeito. Não é necessário
         nenhum peso manual entre as modalidades.

      3. Robustez dimensional: qualquer inconsistência de tamanho entre o
         vetor calculado e as colunas esperadas do dataset_master.csv é
         detectada e barrada ANTES de gravar qualquer arquivo (ver
         `validar_dimensao_vetor` e `carregar_ou_inicializar_dataset`),
         nunca de forma silenciosa.
    """
    if features_audio.ndim != 1 or vetor_texto.ndim != 1:
        raise ExtracaoFeaturesError("Vetores de áudio e de texto devem ser 1-D antes da fusão.")
    vetor = np.concatenate([features_audio, vetor_texto]).astype(np.float64)
    return np.nan_to_num(vetor, nan=0.0, posinf=0.0, neginf=0.0)


def validar_dimensao_vetor(vetor_multimodal: np.ndarray, dim_texto: int) -> None:
    esperado = len(nomes_colunas_features(dim_texto))
    if len(vetor_multimodal) != esperado:
        raise ExtracaoFeaturesError(
            f"Dimensão do vetor multimodal ({len(vetor_multimodal)}) não corresponde "
            f"ao número de colunas de features esperado ({esperado})."
        )


# ==============================================================================
# 8. PERSISTÊNCIA DO DATASET MESTRE (dataset_master.csv)
# ==============================================================================

COLUNAS_META: List[str] = ["UUID", "rotulo", "audio_path", "text_path"]
# Metadados opcionais (datasets antigos, sem essas colunas, continuam funcionando):
#   hash_video       -> SHA-256 do arquivo enviado, usado para barrar duplicatas
#   arquivo_original -> nome do arquivo enviado na interface
#   origem           -> "admin" (aba Alimentação) ou "diagnostico" (confirmação do operador)
#   criado_em        -> data/hora do cadastro
COLUNAS_META_EXTRAS: List[str] = [
    "hash_video", "arquivo_original", "origem", "criado_em",
    "modelo_maquina", "componente_recomendado", "causas_provaveis",
]


def gerar_uuid_curto(df_existente: Optional[pd.DataFrame] = None) -> str:
    """Gera um UUID curto e único no padrão do briefing (ex.: 'ID-8f92a')."""
    existentes = set(df_existente["UUID"]) if df_existente is not None and "UUID" in df_existente.columns else set()
    for _ in range(10):
        candidato = f"ID-{uuid.uuid4().hex[:5]}"
        if candidato not in existentes:
            return candidato
    return f"ID-{uuid.uuid4().hex[:10]}"  # fallback extremamente improvável de colisão


def slugificar_rotulo(rotulo: str) -> str:
    """Normaliza o rótulo para uso seguro em nomes de arquivo: [ROTULO]."""
    limpo = (rotulo or "").strip().upper()
    limpo = "".join(c if c.isalnum() else "_" for c in limpo)
    while "__" in limpo:
        limpo = limpo.replace("__", "_")
    limpo = limpo.strip("_")
    return limpo or "SEM_ROTULO"


def _chave_rotulo(rotulo: str) -> str:
    """Chave de comparação que ignora maiúsculas/minúsculas, acentos e espaços extras."""
    sem_acento = unicodedata.normalize("NFKD", rotulo or "")
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return " ".join(sem_acento.casefold().split())


def normalizar_rotulo(rotulo: str, rotulos_existentes: Sequence[str]) -> str:
    """Evita que "Rolamento", "rolamento " e "ROLAMENTO" virem três classes
    diferentes: se o rótulo digitado equivale a um já cadastrado, reaproveita
    a grafia existente."""
    limpo = " ".join((rotulo or "").split())
    chave = _chave_rotulo(limpo)
    for existente in rotulos_existentes:
        if _chave_rotulo(existente) == chave:
            return existente
    return limpo


def rotulos_do_dataset(df: pd.DataFrame) -> List[str]:
    if df.empty or "rotulo" not in df.columns:
        return []
    return sorted({str(r) for r in df["rotulo"] if str(r).strip()})


def _caminho_relativo(caminho: Path) -> str:
    try:
        return caminho.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(caminho)


def _resolver_caminho(caminho_salvo: str) -> Path:
    """Aceita caminhos relativos gravados em qualquer sistema operacional."""
    caminho = Path(str(caminho_salvo).replace("\\", "/"))
    return caminho if caminho.is_absolute() else BASE_DIR / caminho


def _gravar_atomico(destino: Path, escrever: Callable[[Path], None]) -> None:
    """Escreve num arquivo temporário e só então substitui o destino -- um
    travamento no meio da escrita nunca deixa o arquivo corrompido. Tenta de
    novo por alguns segundos se o arquivo estiver momentaneamente bloqueado
    (ex.: sincronização do OneDrive)."""
    tmp = destino.with_name(f".{destino.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        escrever(tmp)
        for tentativa in range(10):
            try:
                os.replace(tmp, destino)
                return
            except PermissionError:
                if tentativa == 9:
                    raise
                time.sleep(0.3)
    finally:
        tmp.unlink(missing_ok=True)


def carregar_ou_inicializar_dataset(dim_texto: int) -> pd.DataFrame:
    """Carrega o dataset_master.csv (ou cria um novo, vazio, com as colunas
    corretas) e valida a compatibilidade dimensional com o modelo de texto atual."""
    colunas_features = nomes_colunas_features(dim_texto)
    colunas_base = COLUNAS_META + COLUNAS_META_EXTRAS + colunas_features
    vazio = pd.DataFrame(columns=colunas_base)

    if not DATASET_PATH.exists():
        return vazio
    try:
        df = pd.read_csv(DATASET_PATH, dtype={c: str for c in COLUNAS_META + COLUNAS_META_EXTRAS})
    except pd.errors.EmptyDataError:
        return vazio
    except Exception as exc:
        raise DatasetError(f"dataset_master.csv corrompido ou ilegível: {exc}") from exc

    faltantes = [c for c in COLUNAS_META + colunas_features if c not in df.columns]
    if faltantes:
        if df.empty:  # só o cabeçalho, de outra dimensão: não há dado a perder
            logger.warning("dataset_master.csv vazio com cabeçalho incompatível -- recriando o cabeçalho.")
            return vazio
        raise DatasetError(
            "dataset_master.csv não é compatível com a dimensão atual do vetor "
            f"de texto (spaCy). Colunas ausentes: {faltantes[:5]}{'...' if len(faltantes) > 5 else ''}. "
            "Isso costuma acontecer se o modelo spaCy foi trocado após dados já existirem."
        )

    for coluna in COLUNAS_META_EXTRAS:
        if coluna not in df.columns:
            df[coluna] = ""
    for coluna in COLUNAS_META + COLUNAS_META_EXTRAS:
        df[coluna] = df[coluna].fillna("").astype(str)
    outras = [c for c in df.columns if c not in colunas_base]
    return df[colunas_base + outras]


def adicionar_registro_dataset(
    df: pd.DataFrame,
    registro_id: str,
    rotulo: str,
    audio_path: Path,
    text_path: Path,
    vetor_multimodal: np.ndarray,
    dim_texto: int,
    extras: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Insere uma nova linha no dataset mestre, validando a dimensionalidade
    do vetor multimodal ANTES de gravar (prova de falhas dimensionais)."""
    validar_dimensao_vetor(vetor_multimodal, dim_texto)
    nova_linha: Dict[str, Any] = {
        "UUID": registro_id,
        "rotulo": rotulo,
        "audio_path": _caminho_relativo(audio_path),
        "text_path": _caminho_relativo(text_path),
    }
    nova_linha.update({c: str((extras or {}).get(c, "")) for c in COLUNAS_META_EXTRAS})
    nova_linha.update(dict(zip(nomes_colunas_features(dim_texto), vetor_multimodal.tolist())))
    nova = pd.DataFrame([nova_linha]).reindex(columns=df.columns)
    if df.empty:
        return nova
    return pd.concat([df, nova], ignore_index=True)


def salvar_dataset_master(df: pd.DataFrame) -> None:
    try:
        _gravar_atomico(DATASET_PATH, lambda caminho: df.to_csv(caminho, index=False))
    except PermissionError as exc:
        raise DatasetError(
            "não foi possível gravar o dataset_master.csv porque ele está aberto em outro programa "
            "(Excel?) ou bloqueado pela sincronização do OneDrive. Feche o arquivo e tente de novo."
        ) from exc
    except Exception as exc:
        raise DatasetError(f"falha ao salvar dataset_master.csv: {exc}") from exc


def salvar_arquivos_conhecidos(rotulo: str, registro_id: str, wav_bytes: bytes, texto: str) -> Tuple[Path, Path]:
    """Grava o áudio processado e o texto limpo com a nomenclatura
    [ROTULO]_[UUID] nas pastas audios_conhecidos/ e textos_conhecidos/.
    A escrita é feita pelo Python (bytes prontos), que lida bem com pastas
    acentuadas como "Área de Trabalho"."""
    nome_base = f"{slugificar_rotulo(rotulo)}_{registro_id}"
    destino_audio = AUDIOS_DIR / f"{nome_base}.wav"
    destino_texto = TEXTOS_DIR / f"{nome_base}.txt"
    try:
        destino_audio.write_bytes(wav_bytes)
    except Exception as exc:
        raise ExtracaoAudioError(f"falha ao salvar o áudio conhecido '{destino_audio.name}': {exc}") from exc
    try:
        destino_texto.write_text(limpar_texto(texto), encoding="utf-8")
    except Exception as exc:
        destino_audio.unlink(missing_ok=True)
        raise DatasetError(f"falha ao salvar o texto conhecido '{destino_texto.name}': {exc}") from exc
    return destino_audio, destino_texto


def remover_arquivos_caso(audio_path: str, text_path: str) -> None:
    for caminho in (audio_path, text_path):
        if caminho:
            try:
                _resolver_caminho(caminho).unlink(missing_ok=True)
            except Exception:
                logger.warning("Não foi possível apagar o arquivo '%s'.", caminho)


def mapa_hashes(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """hash do vídeo -> (UUID, rótulo) dos casos já cadastrados."""
    if df.empty or "hash_video" not in df.columns:
        return {}
    return {
        str(h): (str(u), str(r))
        for h, u, r in zip(df["hash_video"], df["UUID"], df["rotulo"])
        if str(h).strip()
    }


# ==============================================================================
# 9. TREINO DO MODELO (LightGBM -- Cold Start / Regra de Acumulação)
# ==============================================================================

@dataclass
class ResultadoTreino:
    modelo: LGBMClassifier
    encoder: LabelEncoder
    classes_treinadas: List[str]
    classes_aguardando: Dict[str, int]
    n_amostras_treino: int
    acuracia_cv: Optional[float] = None
    n_folds_cv: int = 0


@dataclass
class EstadoModelo:
    modelo: LGBMClassifier
    encoder: LabelEncoder
    metadata: Dict[str, Any]


def preparar_dados_treino(df_master: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """REGRA DE ACUMULAÇÃO (Seção 3): agrupa por rótulo e mantém no treino
    apenas as classes com >= MIN_SAMPLES_PER_CLASS amostras. As demais linhas
    permanecem intactas no CSV, apenas ficam de fora do LabelEncoder/LightGBM
    até acumularem amostras suficientes."""
    if df_master.empty:
        return df_master.copy(), {}
    contagens = df_master["rotulo"].value_counts()
    classes_validas = contagens[contagens >= MIN_SAMPLES_PER_CLASS].index.tolist()
    classes_aguardando = {str(k): int(v) for k, v in contagens[contagens < MIN_SAMPLES_PER_CLASS].items()}
    df_treino = df_master[df_master["rotulo"].isin(classes_validas)].copy()
    return df_treino, classes_aguardando


def assinatura_dados_treino(df_treino: pd.DataFrame) -> str:
    """Identifica o conjunto exato de casos usado num treino (para saber se o
    modelo salvo está desatualizado em relação ao dataset)."""
    ids = sorted(str(u) for u in df_treino["UUID"]) if not df_treino.empty else []
    return hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()


def pode_treinar(df_treino: pd.DataFrame) -> bool:
    return not df_treino.empty and df_treino["rotulo"].nunique() >= MIN_CLASSES_TREINO


def descrever_pendencias_treino(contagens: Dict[str, int]) -> str:
    """Explica, em linguagem direta, o que falta para o primeiro treino."""
    if not contagens:
        return (
            f"ainda não há casos cadastrados. O primeiro treino exige pelo menos "
            f"{MIN_CLASSES_TREINO} problemas diferentes com {MIN_SAMPLES_PER_CLASS} casos cada."
        )
    prontas = sorted(r for r, n in contagens.items() if n >= MIN_SAMPLES_PER_CLASS)
    faltando = sorted(
        ((r, MIN_SAMPLES_PER_CLASS - n) for r, n in contagens.items() if n < MIN_SAMPLES_PER_CLASS),
        key=lambda kv: (kv[1], kv[0]),
    )
    partes = [
        f"o primeiro treino exige pelo menos {MIN_CLASSES_TREINO} problemas diferentes "
        f"com {MIN_SAMPLES_PER_CLASS} casos cada."
    ]
    if prontas:
        partes.append(f"Já prontos: {', '.join(prontas)}.")
    precisa = MIN_CLASSES_TREINO - len(prontas)
    if faltando:
        partes.append(
            "Faltam: " + "; ".join(f"{r} (+{n} caso{'s' if n > 1 else ''})" for r, n in faltando[:6])
            + ("..." if len(faltando) > 6 else "") + "."
        )
    if len(faltando) < precisa:
        novos = precisa - len(faltando)
        partes.append(f"Cadastre também {novos} problema{'s' if novos > 1 else ''} diferente{'s' if novos > 1 else ''}.")
    return " ".join(partes)


def avaliar_por_validacao_cruzada(X: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], int]:
    """Estima a acurácia em dados NÃO vistos no treino (validação cruzada
    estratificada). Não muda o modelo final -- só informa o quão confiável
    ele tende a ser. Com poucos casos por classe a estimativa é grosseira."""
    menor_classe = int(np.bincount(y).min())
    n_folds = min(N_FOLDS_MAX_CV, menor_classe)
    if n_folds < 2:
        return None, 0
    acertos = 0
    for idx_treino, idx_teste in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X, y):
        modelo = LGBMClassifier(**LGBM_PARAMS)
        modelo.fit(X[idx_treino], y[idx_treino])
        acertos += int((modelo.predict(X[idx_teste]) == y[idx_teste]).sum())
    return acertos / len(y), n_folds


def _dump_atomico(objeto: Any, destino: Path) -> None:
    _gravar_atomico(destino, lambda caminho: joblib.dump(objeto, caminho))


def treinar_modelo(df_treino: pd.DataFrame, dim_texto: int, classes_aguardando: Dict[str, int]) -> ResultadoTreino:
    """Treina o LightGBM com hiperparâmetros FIXOS (sem GridSearchCV) e
    sobrescreve modelo_lgbm.joblib, o LabelEncoder e os metadados em disco."""
    if df_treino.empty:
        raise DatasetError("não há amostras suficientes para treinar (nenhuma classe atingiu o mínimo exigido).")

    colunas_features = nomes_colunas_features(dim_texto)
    faltantes = [c for c in colunas_features if c not in df_treino.columns]
    if faltantes:
        raise ExtracaoFeaturesError(f"colunas de features ausentes no dataset de treino: {faltantes[:5]}")

    X = df_treino[colunas_features].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df_treino["rotulo"].astype(str).to_numpy())

    try:
        acuracia_cv, n_folds = avaliar_por_validacao_cruzada(X, y)
    except Exception:
        logger.exception("Falha na validação cruzada (o treino segue normalmente)")
        acuracia_cv, n_folds = None, 0

    try:
        modelo = LGBMClassifier(**LGBM_PARAMS)
        modelo.fit(X, y)  # treino ultra-rápido: um único .fit(), sem busca de hiperparâmetros
    except Exception as exc:
        raise PipelineError(f"falha ao treinar o LightGBM: {exc}") from exc

    contagem = df_treino["rotulo"].astype(str).value_counts()
    metadata = {
        "colunas_features": colunas_features,
        "dim_texto": dim_texto,
        "classes": encoder.classes_.tolist(),
        "n_amostras_treino": int(len(df_treino)),
        "contagem_por_classe": {str(k): int(v) for k, v in contagem.items()},
        "acuracia_cv": acuracia_cv,
        "n_folds_cv": n_folds,
        "assinatura_dados": assinatura_dados_treino(df_treino),
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        MODELOS_DIR.mkdir(parents=True, exist_ok=True)
        _dump_atomico(modelo, MODEL_PATH)
        _dump_atomico(encoder, ENCODER_PATH)
        _gravar_atomico(
            METADATA_PATH,
            lambda caminho: caminho.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"),
        )
    except Exception as exc:
        raise PipelineError(f"modelo treinado, mas falhou ao salvar em disco: {exc}") from exc

    logger.info("LightGBM treinado: %d amostras, classes=%s, acurácia CV=%s", len(df_treino), metadata["classes"], acuracia_cv)
    return ResultadoTreino(
        modelo=modelo,
        encoder=encoder,
        classes_treinadas=encoder.classes_.tolist(),
        classes_aguardando=classes_aguardando,
        n_amostras_treino=len(df_treino),
        acuracia_cv=acuracia_cv,
        n_folds_cv=n_folds,
    )


def remover_artefatos_modelo() -> None:
    for caminho in (MODEL_PATH, ENCODER_PATH, METADATA_PATH):
        caminho.unlink(missing_ok=True)


def _assinatura_arquivos_modelo() -> Optional[Tuple[int, ...]]:
    caminhos = (MODEL_PATH, ENCODER_PATH, METADATA_PATH)
    if not all(c.exists() for c in caminhos):
        return None
    return tuple(c.stat().st_mtime_ns for c in caminhos)


@st.cache_resource(show_spinner=False, max_entries=2)
def _carregar_modelo_do_disco(assinatura: Tuple[int, ...]) -> EstadoModelo:
    """Só relê os arquivos quando eles mudam (a assinatura são as datas de
    modificação), em vez de a cada clique na interface. Atenção: o parâmetro
    NÃO pode começar com "_", senão o Streamlit o ignora na chave do cache."""
    del assinatura  # usada apenas como chave do cache
    modelo = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return EstadoModelo(modelo=modelo, encoder=encoder, metadata=metadata)


def carregar_modelo_salvo() -> Optional[EstadoModelo]:
    """COLD START: se os artefatos ainda não existirem em disco, retorna None
    (sem levantar exceção) -- a interface apenas avisa que precisa de dados."""
    assinatura = _assinatura_arquivos_modelo()
    if assinatura is None:
        return None
    try:
        return _carregar_modelo_do_disco(assinatura)
    except Exception as exc:
        logger.error("Falha ao carregar o modelo salvo em disco: %s", exc)
        return None


def diagnosticar(vetor_multimodal: np.ndarray, modelo: LGBMClassifier, encoder: LabelEncoder) -> Dict[str, float]:
    """Roda predict_proba() do modelo carregado em memória sobre o vetor multimodal."""
    X = vetor_multimodal.reshape(1, -1)
    n_esperado = getattr(modelo, "n_features_in_", X.shape[1])
    if X.shape[1] != n_esperado:
        raise ModeloIndisponivelError(
            f"o modelo salvo espera {n_esperado} atributos, mas o vetor atual tem {X.shape[1]}. "
            "Retreine a IA pela aba de Alimentação."
        )
    try:
        probs = modelo.predict_proba(X)[0]
    except Exception as exc:
        raise PipelineError(f"falha ao gerar o diagnóstico: {exc}") from exc
    return {str(rotulo): float(p) for rotulo, p in zip(encoder.classes_, probs)}


def _treinar_ou_adiar(df_master: pd.DataFrame, dim_texto: int) -> Tuple[bool, Optional[ResultadoTreino], Optional[str], Dict[str, int]]:
    """Aplica a regra de acumulação sobre o dataset já atualizado e treina o
    LightGBM se houver pelo menos 2 classes com amostras suficientes."""
    df_treino, classes_aguardando = preparar_dados_treino(df_master)
    if not pode_treinar(df_treino):
        contagens = {str(k): int(v) for k, v in df_master["rotulo"].value_counts().items()} if not df_master.empty else {}
        return False, None, descrever_pendencias_treino(contagens), classes_aguardando
    resultado_treino = treinar_modelo(df_treino, dim_texto, classes_aguardando)
    return True, resultado_treino, None, classes_aguardando


# ==============================================================================
# 10. ORQUESTRAÇÃO DE ALTO NÍVEL (usada pelas duas abas da interface)
# ==============================================================================

@dataclass
class ResultadoExtracaoAudio:
    features_audio: np.ndarray
    wav_bytes: bytes
    sample_rate: int
    metadados: AudioMetadata
    ratio_fala: Optional[float]
    demucs_aplicado: bool
    avisos: List[str]
    hash_video: str


def calcular_hash_video(video_bytes: bytes) -> str:
    return hashlib.sha256(video_bytes).hexdigest()


def processar_video_e_extrair(video_bytes: bytes, nome_arquivo: str, etapa: Etapa = _sem_progresso) -> ResultadoExtracaoAudio:
    """Pipeline completo para UM único vídeo: extrai o áudio via ffmpeg +
    tempfile, valida o sinal, roda VAD/Demucs pontualmente e calcula as 56
    features acústicas."""
    sufixo = Path(nome_arquivo).suffix or ".mp4"
    etapa("extraindo o áudio (ffmpeg)", 0.02)
    caminho_wav = extrair_audio_de_video(video_bytes, sufixo)
    try:
        metadados = obter_metadados_audio(caminho_wav)  # Otimização 1
        y, sr = carregar_amostras_audio(caminho_wav)
    finally:
        caminho_wav.unlink(missing_ok=True)
    logger.info(
        "Áudio extraído de '%s': %.2fs @ %dHz (%d canal(is)).",
        nome_arquivo, metadados.duracao_s, metadados.sample_rate, metadados.canais,
    )

    avisos = validar_sinal_audio(y, sr)
    pre = preprocessar_audio_pipeline(y, sr, etapa)
    avisos += pre.avisos

    etapa("calculando as 56 características acústicas", 0.75)
    features_audio = extrair_features_audio(pre.audio, sr)

    buffer = io.BytesIO()
    sf.write(buffer, pre.audio, sr, format="WAV", subtype="PCM_16")

    return ResultadoExtracaoAudio(
        features_audio=features_audio, wav_bytes=buffer.getvalue(), sample_rate=sr,
        metadados=metadados, ratio_fala=pre.ratio_fala, demucs_aplicado=pre.demucs_aplicado,
        avisos=avisos, hash_video=calcular_hash_video(video_bytes),
    )


def registrar_caso_no_dataset(
    df_master: pd.DataFrame,
    rotulo: str,
    features_audio: np.ndarray,
    texto_laudo: str,
    audio_wav_bytes: bytes,
    dim_texto: int,
    extras: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, str, Tuple[Path, Path]]:
    """Gera o UUID, salva os arquivos conhecidos (áudio + texto) e insere uma
    linha no dataset mestre EM MEMÓRIA -- sem treinar. Isolar esta etapa
    permite acumular vários casos (upload em lote) e treinar apenas UMA vez
    no final, em vez de retreinar o LightGBM a cada item processado."""
    vetor_texto = extrair_vetor_texto(texto_laudo)
    vetor_multimodal = montar_vetor_multimodal(features_audio, vetor_texto)
    validar_dimensao_vetor(vetor_multimodal, dim_texto)  # antes de gravar qualquer arquivo

    registro_id = gerar_uuid_curto(df_master)
    destino_audio, destino_texto = salvar_arquivos_conhecidos(rotulo, registro_id, audio_wav_bytes, texto_laudo)
    try:
        df_atualizado = adicionar_registro_dataset(
            df_master, registro_id, rotulo, destino_audio, destino_texto, vetor_multimodal, dim_texto,
            extras={"criado_em": datetime.now().isoformat(timespec="seconds"), **(extras or {})},
        )
    except Exception:
        remover_arquivos_caso(str(destino_audio), str(destino_texto))
        raise
    return df_atualizado, registro_id, (destino_audio, destino_texto)


def _persistir_novo_caso(
    df_master: pd.DataFrame,
    rotulo: str,
    features_audio: np.ndarray,
    texto_laudo: str,
    audio_wav_bytes: bytes,
    dim_texto: int,
    extras: Dict[str, str],
) -> Tuple[pd.DataFrame, str]:
    """Registra o caso E grava o CSV na hora. Se a gravação do CSV falhar, os
    arquivos recém-criados são apagados (nada de áudios órfãos)."""
    df_novo, registro_id, (audio, texto) = registrar_caso_no_dataset(
        df_master, rotulo, features_audio, texto_laudo, audio_wav_bytes, dim_texto, extras
    )
    try:
        salvar_dataset_master(df_novo)
    except Exception:
        remover_arquivos_caso(str(audio), str(texto))
        raise
    return df_novo, registro_id


@dataclass
class EntradaCaso:
    """Um bloco "Caso N" da aba Admin: um ou mais vídeos do MESMO problema,
    com o mesmo laudo e o mesmo rótulo."""
    caso_id: str
    numero: int
    arquivos: List[Any]  # UploadedFile(s) do Streamlit (ou qualquer objeto com .name e .getvalue())
    texto: str
    rotulo: str
    modelo_maquina: str
    componente_recomendado: str
    causas_provaveis: str


@dataclass
class ItemLote:
    caso_id: str
    numero_caso: int
    arquivo: str
    rotulo: str
    modelo_maquina: str
    total_no_caso: int
    status: str = "pendente"  # "ok" | "duplicado" | "erro"
    registro_id: Optional[str] = None
    duplicado_de: Optional[str] = None
    erro: Optional[str] = None
    duracao_s: Optional[float] = None
    ratio_fala: Optional[float] = None
    demucs_aplicado: bool = False
    avisos: List[str] = field(default_factory=list)


@dataclass
class ResultadoLoteAdmin:
    itens: List[ItemLote]
    treinou: bool
    resultado_treino: Optional[ResultadoTreino] = None
    motivo_sem_treino: Optional[str] = None
    classes_aguardando: Dict[str, int] = field(default_factory=dict)
    erro_treino: Optional[str] = None

    @property
    def n_sucesso(self) -> int:
        return sum(1 for item in self.itens if item.status == "ok")

    @property
    def n_falha(self) -> int:
        return sum(1 for item in self.itens if item.status == "erro")

    def como_dict(self) -> Dict[str, Any]:
        """Versão "leve" (sem o modelo) para guardar no session_state."""
        rt = self.resultado_treino
        return {
            "itens": [asdict(item) for item in self.itens],
            "treinou": self.treinou,
            "classes_treinadas": rt.classes_treinadas if rt else [],
            "acuracia_cv": rt.acuracia_cv if rt else None,
            "n_folds_cv": rt.n_folds_cv if rt else 0,
            "n_amostras_treino": rt.n_amostras_treino if rt else 0,
            "motivo_sem_treino": self.motivo_sem_treino,
            "classes_aguardando": dict(self.classes_aguardando),
            "erro_treino": self.erro_treino,
            "interrompido": False,
        }


def validar_entradas_admin(entradas: Sequence[EntradaCaso]) -> List[str]:
    """Checa TODOS os blocos antes de começar o processamento (que pode ser
    demorado): é melhor avisar já que falta um rótulo do que depois de minutos."""
    problemas: List[str] = []
    for entrada in entradas:
        faltas = []
        if not entrada.arquivos:
            faltas.append("envie ao menos um vídeo")
        if not limpar_texto(entrada.texto):
            faltas.append("preencha o texto do laudo")
        if not limpar_texto(entrada.rotulo):
            faltas.append("informe o nome exato do problema (rótulo)")
        if not limpar_texto(entrada.modelo_maquina):
            faltas.append("informe o modelo da máquina")
        if not limpar_texto(entrada.componente_recomendado):
            faltas.append("informe a peça ou conjunto prioritário para inspeção")
        if not limpar_texto(entrada.causas_provaveis):
            faltas.append("informe as possíveis causas/origens do problema")
        if faltas:
            problemas.append(f"Caso {entrada.numero}: {', '.join(faltas)}.")
    return problemas


def alimentar_sistema_em_lote(
    entradas: Sequence[EntradaCaso],
    progresso: Optional[Callable[[float, str], None]] = None,
    ao_processar_item: Optional[Callable[[ItemLote], None]] = None,
) -> ResultadoLoteAdmin:
    """ABA ADMIN: processa VÁRIOS vídeos de uma vez -- extrai o áudio e as
    features de cada um, registra cada caso no dataset_master.csv assim que
    fica pronto (nada se perde se o envio for interrompido) e, só no final,
    dispara UM ÚNICO retreino do LightGBM sobre o dataset atualizado."""
    progresso = progresso or (lambda _f, _t: None)
    tarefas = [(entrada, arquivo) for entrada in entradas for arquivo in entrada.arquivos]
    total = max(1, len(tarefas))

    with _trava_processamento():
        dim_texto = obter_dimensao_vetor_texto()
        df_master = carregar_ou_inicializar_dataset(dim_texto)
        hashes = mapa_hashes(df_master)

        itens: List[ItemLote] = []
        for k, (entrada, arquivo) in enumerate(tarefas):
            nome = str(getattr(arquivo, "name", f"arquivo_{k + 1}"))
            item = ItemLote(
                caso_id=entrada.caso_id, numero_caso=entrada.numero, arquivo=nome,
                rotulo=limpar_texto(entrada.rotulo), modelo_maquina=limpar_texto(entrada.modelo_maquina),
                total_no_caso=len(entrada.arquivos),
            )

            def etapa(texto: str, fracao: float, _k: int = k, _nome: str = nome) -> None:
                progresso((_k + fracao) / total, f"Vídeo {_k + 1} de {len(tarefas)} · {_nome} — {texto}...")

            try:
                if not (limpar_texto(entrada.texto) and item.rotulo and item.modelo_maquina
                        and limpar_texto(entrada.componente_recomendado) and limpar_texto(entrada.causas_provaveis)):
                    raise PipelineError("o caso precisa de laudo, rótulo, modelo da máquina, peça e possíveis causas.")
                video_bytes = arquivo.getvalue()
                hash_video = calcular_hash_video(video_bytes)
                if hash_video in hashes:
                    item.status = "duplicado"
                    item.registro_id, item.rotulo = hashes[hash_video]
                    item.duplicado_de = item.registro_id
                else:
                    item.rotulo = normalizar_rotulo(item.rotulo, rotulos_do_dataset(df_master))
                    resultado_audio = processar_video_e_extrair(video_bytes, nome, etapa)
                    item.duracao_s = resultado_audio.metadados.duracao_s
                    item.ratio_fala = resultado_audio.ratio_fala
                    item.demucs_aplicado = resultado_audio.demucs_aplicado
                    item.avisos = list(resultado_audio.avisos)
                    if item.duracao_s > MAX_VIDEO_DURATION_WARNING_S:
                        item.avisos.append(
                            f"vídeo com {item.duracao_s / 60:.1f} min, acima dos "
                            f"{MAX_VIDEO_DURATION_WARNING_S // 60} min recomendados; considere cortar da próxima vez."
                        )
                    etapa("salvando na base de conhecimento", 0.95)
                    df_master, registro_id = _persistir_novo_caso(
                        df_master, item.rotulo, resultado_audio.features_audio, entrada.texto,
                        resultado_audio.wav_bytes, dim_texto,
                        extras={
                            "hash_video": hash_video, "arquivo_original": nome, "origem": "admin",
                            "modelo_maquina": item.modelo_maquina,
                            "componente_recomendado": limpar_texto(entrada.componente_recomendado),
                            "causas_provaveis": limpar_texto(entrada.causas_provaveis),
                        },
                    )
                    hashes[hash_video] = (registro_id, item.rotulo)
                    item.registro_id = registro_id
                    item.status = "ok"
            except PipelineError as exc:
                item.status, item.erro = "erro", str(exc)
            except Exception as exc:  # defesa extra: um item ruim não deve derrubar o lote inteiro
                logger.exception("Erro inesperado ao processar '%s' no lote de alimentação", nome)
                item.status, item.erro = "erro", f"erro inesperado: {exc}"
            itens.append(item)
            if ao_processar_item is not None:
                ao_processar_item(item)

        resultado = ResultadoLoteAdmin(itens=itens, treinou=False)
        if not any(item.status == "ok" for item in itens):
            resultado.motivo_sem_treino = "nenhum vídeo novo foi cadastrado neste envio."
            return resultado

        progresso(1.0, "Treinando a IA (LightGBM)...")
        try:
            treinou, resultado_treino, motivo, aguardando = _treinar_ou_adiar(df_master, dim_texto)
            resultado.treinou, resultado.resultado_treino = treinou, resultado_treino
            resultado.motivo_sem_treino, resultado.classes_aguardando = motivo, aguardando
        except PipelineError as exc:
            resultado.erro_treino = str(exc)
        return resultado


@dataclass
class ResultadoAbsorcao:
    registro_id: Optional[str]
    rotulo: str
    treinou: bool
    resultado_treino: Optional[ResultadoTreino] = None
    motivo_sem_treino: Optional[str] = None
    classes_aguardando: Dict[str, int] = field(default_factory=dict)
    duplicado_de: Optional[str] = None


def absorver_caso(
    rotulo: str,
    features_audio: np.ndarray,
    texto_laudo: str,
    audio_wav_bytes: bytes,
    hash_video: str = "",
    nome_arquivo: str = "",
) -> ResultadoAbsorcao:
    """Passo final usado pela aba de Diagnóstico: gera UUID, salva os arquivos
    conhecidos, insere no dataset_master.csv e RETREINA a IA respeitando a
    regra de acumulação (>= MIN_SAMPLES_PER_CLASS amostras por classe)."""
    with _trava_processamento():
        dim_texto = obter_dimensao_vetor_texto()
        df_master = carregar_ou_inicializar_dataset(dim_texto)
        rotulo_final = normalizar_rotulo(rotulo, rotulos_do_dataset(df_master))

        hashes = mapa_hashes(df_master)
        if hash_video and hash_video in hashes:
            registro_existente, rotulo_existente = hashes[hash_video]
            return ResultadoAbsorcao(
                registro_id=registro_existente, rotulo=rotulo_existente, treinou=False,
                duplicado_de=registro_existente,
                motivo_sem_treino="este vídeo já estava cadastrado na base, então não foi adicionado de novo.",
            )

        df_master, registro_id = _persistir_novo_caso(
            df_master, rotulo_final, features_audio, texto_laudo, audio_wav_bytes, dim_texto,
            extras={"hash_video": hash_video, "arquivo_original": nome_arquivo, "origem": "diagnostico"},
        )
        treinou, resultado_treino, motivo_sem_treino, classes_aguardando = _treinar_ou_adiar(df_master, dim_texto)
        return ResultadoAbsorcao(
            registro_id=registro_id, rotulo=rotulo_final, treinou=treinou,
            resultado_treino=resultado_treino, motivo_sem_treino=motivo_sem_treino,
            classes_aguardando=classes_aguardando,
        )


def gerar_diagnostico(
    video_bytes: bytes,
    nome_arquivo: str,
    texto_laudo: str,
    modelo_maquina: str,
    estado_modelo: Optional[EstadoModelo],
    etapa: Etapa = _sem_progresso,
) -> Dict[str, Any]:
    """ABA DIAGNÓSTICO -- Passo A: extrai features do vídeo/texto desconhecidos
    e chama predict_proba() do modelo carregado. Devolve as probabilidades e
    os dados brutos, para reaproveitar sem reprocessar o vídeo quando o
    operador confirmar (ou corrigir) o problema."""
    if estado_modelo is None:
        raise ModeloIndisponivelError(
            "ainda não existe um modelo treinado. Alimente o sistema pela aba 'Alimentação (Admin)' primeiro."
        )
    if not video_bytes:
        raise PipelineError("envie um vídeo desconhecido antes de gerar o diagnóstico.")
    if not limpar_texto(texto_laudo):
        raise PipelineError("o texto do laudo não pode estar vazio.")
    if not limpar_texto(modelo_maquina):
        raise PipelineError("informe o modelo da máquina para gerar um relatório direcionado.")

    with _trava_processamento():
        resultado_audio = processar_video_e_extrair(video_bytes, nome_arquivo, etapa)
        etapa("calculando as probabilidades", 0.95)
        vetor_texto = extrair_vetor_texto(texto_laudo)
        vetor_multimodal = montar_vetor_multimodal(resultado_audio.features_audio, vetor_texto)
        probs = diagnosticar(vetor_multimodal, estado_modelo.modelo, estado_modelo.encoder)

    return {
        "probs": probs,
        "features_audio": resultado_audio.features_audio.tolist(),
        "texto_laudo": texto_laudo,
        "modelo_maquina": limpar_texto(modelo_maquina),
        "wav_bytes": resultado_audio.wav_bytes,
        "hash_video": resultado_audio.hash_video,
        "nome_arquivo": nome_arquivo,
        "metadados": asdict(resultado_audio.metadados),
        "ratio_fala": resultado_audio.ratio_fala,
        "demucs_aplicado": resultado_audio.demucs_aplicado,
        "avisos": list(resultado_audio.avisos),
    }


def obter_referencias_relatorio(rotulo: str, modelo_maquina: str) -> Dict[str, Any]:
    """Busca recomendações registradas pelos administradores para o relatório.

    Primeiro usa casos do mesmo problema e modelo; se a base ainda não possuir
    essa combinação, usa o histórico do problema e sinaliza a menor precisão.
    """
    dim_texto = obter_dimensao_vetor_texto()
    df = carregar_ou_inicializar_dataset(dim_texto)
    if df.empty:
        return {"casos": 0, "especifico_modelo": False, "pecas": [], "causas": []}

    mesmos_problema = df[df["rotulo"].astype(str) == str(rotulo)]
    mesma_maquina = mesmos_problema[
        mesmos_problema["modelo_maquina"].map(_chave_rotulo) == _chave_rotulo(modelo_maquina)
    ]
    referencias = mesma_maquina if not mesma_maquina.empty else mesmos_problema

    def valores_unicos(coluna: str) -> List[str]:
        valores = [limpar_texto(v) for v in referencias.get(coluna, pd.Series(dtype=str)).tolist()]
        return list(dict.fromkeys(v for v in valores if v))

    return {
        "casos": int(len(referencias)),
        "especifico_modelo": not mesma_maquina.empty,
        "pecas": valores_unicos("componente_recomendado"),
        "causas": valores_unicos("causas_provaveis"),
    }


def interpretar_confianca(probabilidade: float, segunda_probabilidade: float) -> Tuple[str, str]:
    if probabilidade >= 0.80:
        return "Indicação consistente", "Confirme a indicação com inspeção física antes de intervir."
    if probabilidade >= 0.50 and probabilidade - segunda_probabilidade >= 0.10:
        return "Indicação moderada", "Há uma hipótese principal, mas a inspeção deve comparar alternativas próximas."
    return "Indicação inconclusiva", "Verifique o equipamento antes de decidir; pode haver uma falha ainda não cadastrada."


def render_relatorio_diagnostico(resultado: Dict[str, Any]) -> None:
    """Apresenta um relatório operacional baseado na probabilidade e no histórico."""
    ordenadas = sorted(resultado["probs"].items(), key=lambda item: item[1], reverse=True)
    if not ordenadas:
        st.warning("O modelo não retornou hipóteses para este diagnóstico.")
        return
    rotulo, probabilidade = ordenadas[0]
    segunda_probabilidade = ordenadas[1][1] if len(ordenadas) > 1 else 0.0
    nivel, orientacao = interpretar_confianca(probabilidade, segunda_probabilidade)
    referencias = obter_referencias_relatorio(rotulo, resultado["modelo_maquina"])

    with st.container(border=True):
        st.markdown("### Relatório de diagnóstico")
        st.caption(f"Modelo informado: **{_md(resultado['modelo_maquina'])}**")
        st.markdown(f"**Hipótese principal:** {_md(rotulo)} · **{probabilidade:.0%}** de confiança interna")
        st.info(f"**{nivel}.** {orientacao}")

        if referencias["casos"] and referencias["especifico_modelo"]:
            st.success(
                f"A recomendação abaixo considera {referencias['casos']} caso(s) histórico(s) "
                "do mesmo problema e modelo de máquina."
            )
        elif referencias["casos"]:
            st.warning(
                f"Há {referencias['casos']} caso(s) histórico(s) para este problema, mas nenhum para o "
                "modelo informado. Use as recomendações como referência geral."
            )
        else:
            st.warning(
                "Ainda não há referência técnica cadastrada para esta hipótese. Registre o caso na área "
                "administrativa após a confirmação para enriquecer relatórios futuros."
            )

        st.markdown("**Peças ou conjuntos prioritários para inspeção**")
        if referencias["pecas"]:
            for peca in referencias["pecas"]:
                st.markdown(f"- {_md(peca)}")
        else:
            st.caption("Nenhuma peça ou conjunto foi associado a esta hipótese na base de conhecimento.")

        st.markdown("**Possíveis origens do problema**")
        if referencias["causas"]:
            for causa in referencias["causas"]:
                st.markdown(f"- {_md(causa)}")
        else:
            st.caption("Nenhuma causa foi registrada para esta hipótese. Faça a inspeção seguindo o procedimento técnico da máquina.")

        st.markdown("**Próximos passos sugeridos**")
        st.markdown(
            "1. Inspecione os componentes listados e compare com o laudo e o áudio.\n"
            "2. Verifique condições de montagem, desgaste, contaminação, obstrução e operação.\n"
            "3. Confirme a falha antes de substituir peças ou liberar o equipamento."
        )

    linhas = [
        "RELATÓRIO DE DIAGNÓSTICO ACÚSTICO", "",
        f"Modelo da máquina: {resultado['modelo_maquina']}",
        f"Hipótese principal: {rotulo}", f"Confiança interna: {probabilidade:.0%}",
        f"Classificação: {nivel}", "", "Peças/conjuntos prioritários:",
        *(referencias["pecas"] or ["Sem referência cadastrada."]), "", "Possíveis origens:",
        *(referencias["causas"] or ["Sem referência cadastrada."]), "", "Importante: confirme o resultado por inspeção técnica.",
    ]
    st.download_button(
        "Baixar relatório (.txt)", data="\n".join(linhas),
        file_name="relatorio_diagnostico.txt", mime="text/plain", width="stretch",
    )


def excluir_casos(uuids: Sequence[str]) -> Dict[str, Any]:
    """Remove casos do dataset (e seus arquivos de áudio/laudo) e retreina.
    Se depois da exclusão não houver dados suficientes, o modelo salvo é
    apagado -- ele conteria classes cujos exemplos não existem mais."""
    with _trava_processamento():
        dim_texto = obter_dimensao_vetor_texto()
        df_master = carregar_ou_inicializar_dataset(dim_texto)
        mascara = df_master["UUID"].isin(list(uuids))
        removidos = df_master[mascara]
        df_restante = df_master[~mascara].reset_index(drop=True)
        salvar_dataset_master(df_restante)
        for audio, texto in zip(removidos["audio_path"], removidos["text_path"]):
            remover_arquivos_caso(audio, texto)

        treinou, resultado_treino, motivo, _ = _treinar_ou_adiar(df_restante, dim_texto)
        if not treinou:
            remover_artefatos_modelo()
        return {
            "n_removidos": int(mascara.sum()),
            "treinou": treinou,
            "classes_treinadas": resultado_treino.classes_treinadas if resultado_treino else [],
            "motivo_sem_treino": motivo,
        }


def retreinar_agora() -> Tuple[bool, Optional[ResultadoTreino], Optional[str]]:
    """Treina a IA sobre o dataset atual (usado pelo botão "Treinar IA agora")."""
    with _trava_processamento():
        dim_texto = obter_dimensao_vetor_texto()
        df_master = carregar_ou_inicializar_dataset(dim_texto)
        treinou, resultado_treino, motivo, _ = _treinar_ou_adiar(df_master, dim_texto)
        return treinou, resultado_treino, motivo


@dataclass
class StatusBase:
    """Fotografia da base de conhecimento, usada para os painéis da interface."""
    total_casos: int = 0
    contagens: Dict[str, int] = field(default_factory=dict)
    pode_treinar: bool = False
    modelo_desatualizado: bool = False
    motivo_sem_treino: Optional[str] = None
    tabela_casos: Optional[pd.DataFrame] = None
    erro: Optional[str] = None


@st.cache_data(show_spinner=False, max_entries=4)
def _ler_dataset_cacheado(assinatura: Tuple[int, int], dim_texto: int) -> pd.DataFrame:
    """Leitura do CSV reaproveitada entre cliques enquanto o arquivo não muda
    (a assinatura -- data de modificação e tamanho -- é a chave do cache)."""
    del assinatura
    return carregar_ou_inicializar_dataset(dim_texto)


def obter_status_base(estado_modelo: Optional[EstadoModelo]) -> StatusBase:
    try:
        dim_texto = obter_dimensao_vetor_texto()
        if DATASET_PATH.exists():
            info = DATASET_PATH.stat()
            df = _ler_dataset_cacheado((info.st_mtime_ns, info.st_size), dim_texto)
        else:
            df = carregar_ou_inicializar_dataset(dim_texto)
    except PipelineError as exc:
        return StatusBase(erro=str(exc))
    except Exception as exc:
        logger.exception("Falha ao ler o status da base")
        return StatusBase(erro=f"falha ao ler a base de conhecimento: {exc}")

    contagens = {str(k): int(v) for k, v in df["rotulo"].value_counts().items()} if not df.empty else {}
    df_treino, _ = preparar_dados_treino(df)
    treinavel = pode_treinar(df_treino)
    assinatura_modelo = estado_modelo.metadata.get("assinatura_dados") if estado_modelo else None
    desatualizado = treinavel and assinatura_modelo != assinatura_dados_treino(df_treino)
    return StatusBase(
        total_casos=len(df),
        contagens=contagens,
        pode_treinar=treinavel,
        modelo_desatualizado=desatualizado,
        motivo_sem_treino=None if treinavel else descrever_pendencias_treino(contagens),
        tabela_casos=df[COLUNAS_META + COLUNAS_META_EXTRAS].copy(),
    )


# ==============================================================================
# 11. BACKUP / MIGRAÇÃO ENTRE MÁQUINAS
# ==============================================================================

def gerar_backup_zip() -> bytes:
    """Empacota tudo que a IA já aprendeu -- dataset_master.csv, os áudios e
    textos conhecidos e o modelo LightGBM treinado -- em um único .zip. Basta
    baixar esse arquivo e extraí-lo na pasta do app.py em outro computador
    para continuar exatamente de onde parou, sem retreinar do zero.

    Só é chamado quando o usuário clica no botão de download (antes, o .zip
    inteiro era montado a cada clique em qualquer lugar do app)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if DATASET_PATH.exists():
            zf.write(DATASET_PATH, arcname=DATASET_PATH.name)
        for pasta in (AUDIOS_DIR, TEXTOS_DIR, MODELOS_DIR):
            if not pasta.exists():
                continue
            for arquivo in pasta.rglob("*"):
                if arquivo.is_file() and not arquivo.name.endswith(".tmp"):
                    # WAV quase não comprime; guardar sem compressão deixa o backup muito mais rápido.
                    compressao = zipfile.ZIP_STORED if arquivo.suffix.lower() == ".wav" else zipfile.ZIP_DEFLATED
                    zf.write(arquivo, arcname=arquivo.relative_to(BASE_DIR).as_posix(), compress_type=compressao)
    return buffer.getvalue()


def verificar_ambiente() -> List[Tuple[str, bool, str]]:
    """Checagem rápida das dependências externas, exibida na barra lateral."""
    itens: List[Tuple[str, bool, str]] = []
    ffmpeg = localizar_ffmpeg()
    if ffmpeg is None:
        detalhe_ffmpeg = "não encontrado — o áudio dos vídeos não pode ser extraído"
    elif "imageio_ffmpeg" in ffmpeg:
        detalhe_ffmpeg = "usando o ffmpeg incluído no pacote"
    else:
        detalhe_ffmpeg = "encontrado no sistema"
    itens.append(("ffmpeg", ffmpeg is not None, detalhe_ffmpeg))
    try:
        carregar_spacy()
        itens.append(("spaCy (laudos)", True, f"{SPACY_MODEL_NAME} carregado"))
    except PipelineError as exc:
        itens.append(("spaCy (laudos)", False, str(exc)))
    try:
        carregar_vad()
        itens.append(("Silero VAD (voz)", True, "carregado"))
    except PipelineError as exc:
        itens.append(("Silero VAD (voz)", False, f"{exc} — os vídeos serão usados sem remoção de voz"))
    if importlib.util.find_spec("demucs") is not None:
        itens.append(("Demucs (remoção de voz)", True, "instalado — o modelo é baixado no primeiro uso"))
    else:
        itens.append(("Demucs (remoção de voz)", False, "não instalado — a voz não será removida"))
    return itens


# ==============================================================================
# 12. INTERFACE STREAMLIT
# ==============================================================================

_MD_ESPECIAIS = set("\\`*_[]<>#|~$")


def _md(texto: Any) -> str:
    """Escapa texto do usuário (nomes de arquivo, rótulos) antes de usá-lo em
    markdown -- ex.: "video_01_rolamento.mp4" não pode virar itálico."""
    return "".join("\\" + c if c in _MD_ESPECIAIS else c for c in str(texto))


def _formatar_data(iso: Optional[str]) -> str:
    try:
        return datetime.fromisoformat(str(iso)).strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError):
        return "—"


def _novo_id_caso() -> str:
    return uuid.uuid4().hex[:8]


def inicializar_sessao() -> None:
    padroes: Dict[str, Any] = {
        # Aba Admin: cada bloco "Caso N" tem um id estável, usado nas chaves dos
        # widgets. Remover um id faz o Streamlit descartar os widgets daquele
        # bloco -- inclusive os arquivos do file_uploader, que não podem ser
        # limpos de outra forma.
        "admin_casos": [_novo_id_caso()],
        "admin_erros_casos": {},
        "admin_resultado": None,        # resumo do último envio (fica na tela até o próximo)
        "admin_parcial": None,          # itens já processados de um envio em andamento
        "admin_toast_pendente": False,
        "admin_tabela_geracao": 0,
        "admin_autenticado": False,
        # Aba Diagnóstico
        "diag_resultado": None,
        "diag_geracao": 0,
        # Mensagens exibidas uma única vez depois de um st.rerun()
        "flash": {},
    }
    for chave, valor in padroes.items():
        st.session_state.setdefault(chave, valor)


def obter_senha_admin() -> str:
    """Lê a senha fora do repositório.

    Prioridade: ``.streamlit/secrets.toml`` (admin_password) e, como alternativa
    para implantação, a variável de ambiente ``DIAGNOSTICO_ADMIN_PASSWORD``.
    """
    try:
        senha = str(st.secrets.get("admin_password", ""))
    except Exception:
        senha = ""
    return senha or os.getenv("DIAGNOSTICO_ADMIN_PASSWORD", "")


def render_acesso_admin() -> bool:
    """Exibe a autenticação antes de qualquer controle de alteração da base."""
    if st.session_state.get("admin_autenticado"):
        if st.button("Sair da área administrativa", key="admin_sair"):
            st.session_state["admin_autenticado"] = False
            st.rerun()
        return True

    st.subheader("Área administrativa")
    senha_configurada = obter_senha_admin()
    if not senha_configurada:
        st.error(
            "A senha administrativa ainda não foi configurada. Um responsável deve criar "
            "`.streamlit/secrets.toml` a partir de `.streamlit/secrets.example.toml` "
            "ou definir a variável `DIAGNOSTICO_ADMIN_PASSWORD`."
        )
        return False

    st.caption("Apenas funcionários autorizados podem cadastrar, excluir ou treinar dados.")
    with st.form("form_acesso_admin", clear_on_submit=True):
        senha_informada = st.text_input("Senha administrativa", type="password")
        entrar = st.form_submit_button("Acessar administração")
    if entrar:
        if hmac.compare_digest(senha_informada, senha_configurada):
            st.session_state["admin_autenticado"] = True
            st.rerun()
        st.error("Senha incorreta.")
    return False


def _flash(area: str, tipo: str, texto: str, toast: Optional[str] = None) -> None:
    """Guarda uma mensagem para ser exibida DEPOIS do st.rerun(). Sem isso, a
    confirmação some no mesmo instante em que aparece (era o que acontecia)."""
    st.session_state["flash"].setdefault(area, []).append({"tipo": tipo, "texto": texto, "toast": toast})


def _mostrar_flash(area: str) -> None:
    for mensagem in st.session_state["flash"].pop(area, []):
        getattr(st, mensagem["tipo"])(mensagem["texto"])
        if mensagem.get("toast"):
            st.toast(mensagem["toast"])


def _descricao_modelo(metadata: Dict[str, Any]) -> str:
    partes = [f"treinada em {_formatar_data(metadata.get('treinado_em'))}"]
    if metadata.get("n_amostras_treino"):
        partes.append(f"{metadata['n_amostras_treino']} casos")
    acuracia = metadata.get("acuracia_cv")
    if acuracia is not None:
        partes.append(
            f"acurácia estimada {acuracia:.0%} (validação cruzada em {metadata.get('n_folds_cv', '?')} partes)"
        )
    return " · ".join(partes)


# ---------------------------------------------------------------------- Admin

def _adicionar_caso_admin() -> None:
    st.session_state["admin_casos"].append(_novo_id_caso())


def _limpar_widgets_caso(caso_id: str) -> None:
    for prefixo in (
        "admin_video_", "admin_texto_", "admin_rotulo_", "admin_modelo_",
        "admin_componente_", "admin_causas_", "admin_remover_",
    ):
        st.session_state.pop(f"{prefixo}{caso_id}", None)
    st.session_state["admin_erros_casos"].pop(caso_id, None)


def _remover_caso_admin(caso_id: str) -> None:
    casos = st.session_state["admin_casos"]
    if caso_id in casos and len(casos) > 1:
        casos.remove(caso_id)
        _limpar_widgets_caso(caso_id)


def _fechar_resultado_admin() -> None:
    st.session_state["admin_resultado"] = None


def _atualizar_casos_apos_envio(itens: List[Dict[str, Any]]) -> None:
    """Some com os blocos cujos vídeos foram todos cadastrados (ou já estavam
    na base) e mantém na tela apenas os que tiveram erro, com o motivo, para
    o usuário corrigir e reenviar. Sempre sobra ao menos um bloco vazio."""
    por_caso: Dict[str, List[Dict[str, Any]]] = {}
    for item in itens:
        por_caso.setdefault(item["caso_id"], []).append(item)

    manter: List[str] = []
    erros: Dict[str, List[str]] = {}
    for caso_id in st.session_state["admin_casos"]:
        itens_caso = por_caso.get(caso_id)
        if not itens_caso:  # bloco que não participou do envio (vazio)
            manter.append(caso_id)
            continue
        falhas = [i for i in itens_caso if i["status"] == "erro"]
        completos = len(itens_caso) >= itens_caso[0]["total_no_caso"]
        if falhas or not completos:
            manter.append(caso_id)
            erros[caso_id] = [f"{i['arquivo']}: {i['erro']}" for i in falhas]
        else:
            _limpar_widgets_caso(caso_id)

    st.session_state["admin_casos"] = manter or [_novo_id_caso()]
    st.session_state["admin_erros_casos"] = {k: v for k, v in erros.items() if v}


def _recuperar_envio_interrompido() -> None:
    """Se um envio foi interrompido no meio (ex.: o usuário clicou em outro
    botão durante o processamento), os vídeos concluídos até ali já estão
    salvos; aqui eles viram o "resultado do último envio"."""
    parcial = st.session_state.get("admin_parcial")
    st.session_state["admin_parcial"] = None
    if not parcial:
        return
    st.session_state["admin_resultado"] = {
        "itens": parcial, "treinou": False, "classes_treinadas": [], "acuracia_cv": None,
        "n_folds_cv": 0, "n_amostras_treino": 0, "classes_aguardando": {}, "erro_treino": None,
        "motivo_sem_treino": None, "interrompido": True,
    }
    st.session_state["admin_toast_pendente"] = True
    _atualizar_casos_apos_envio(parcial)


def _executar_alimentacao(entradas: List[EntradaCaso]) -> None:
    st.session_state["admin_resultado"] = None
    st.session_state["admin_parcial"] = []
    barra = st.progress(0.0, text="Preparando o processamento...")

    def progresso(fracao: float, texto: str) -> None:
        barra.progress(float(min(max(fracao, 0.0), 1.0)), text=_md(texto))

    def ao_processar_item(item: ItemLote) -> None:
        st.session_state["admin_parcial"].append(asdict(item))

    try:
        resultado = alimentar_sistema_em_lote(entradas, progresso=progresso, ao_processar_item=ao_processar_item)
    except PipelineError as exc:
        barra.empty()
        _recuperar_envio_interrompido()
        st.error(f"Não foi possível alimentar o sistema: {exc}")
        return
    except Exception as exc:
        logger.exception("Erro inesperado ao alimentar o sistema em lote")
        barra.empty()
        _recuperar_envio_interrompido()
        st.error(f"Ocorreu um erro inesperado ao alimentar o sistema: {exc}")
        return

    st.session_state["admin_parcial"] = None
    st.session_state["admin_resultado"] = resultado.como_dict()
    st.session_state["admin_toast_pendente"] = True
    _atualizar_casos_apos_envio(st.session_state["admin_resultado"]["itens"])
    st.rerun()


def _linha_item(item: Dict[str, Any]) -> str:
    # Sem "Caso N": depois do envio os blocos são renumerados, e o número confundiria.
    partes = [f"{_md(item['arquivo'])} → **{_md(item['rotulo'])}**"]
    if item.get("duracao_s") is not None:
        partes.append(f"{item['duracao_s']:.1f} s de áudio")
    ratio = item.get("ratio_fala")
    if ratio is not None:
        if item.get("demucs_aplicado"):
            partes.append(f"voz humana em {ratio:.0%} do áudio, removida")
        elif ratio >= MIN_SPEECH_RATIO:
            partes.append(f"voz humana em {ratio:.0%} do áudio (não removida)")
        else:
            partes.append("sem voz humana")
    if item.get("registro_id"):
        partes.append(str(item["registro_id"]))
    return " · ".join(partes)


def render_resultado_admin() -> None:
    """Confirmação persistente do último envio: fica visível até o próximo
    envio (ou até o usuário fechar), e também aparece como notificação."""
    resultado = st.session_state.get("admin_resultado")
    if not resultado:
        return

    itens = resultado["itens"]
    ok = [i for i in itens if i["status"] == "ok"]
    duplicados = [i for i in itens if i["status"] == "duplicado"]
    falhas = [i for i in itens if i["status"] == "erro"]

    if st.session_state.get("admin_toast_pendente"):
        st.session_state["admin_toast_pendente"] = False
        if ok:
            st.toast(f"{len(ok)} vídeo(s) recebido(s) e salvo(s) na base.")
        elif falhas:
            st.toast("Nenhum vídeo foi cadastrado — veja os detalhes abaixo.")
        elif duplicados:
            st.toast("Os vídeos enviados já estavam na base.")

    with st.container(border=True):
        st.markdown("**Resultado do último envio**")
        if resultado.get("interrompido"):
            st.warning(
                "O envio foi interrompido antes de terminar. Os vídeos listados abaixo como "
                "cadastrados já estão salvos; reenvie os demais (duplicatas são ignoradas)."
            )

        if ok:
            st.success(f"{len(ok)} vídeo(s) recebido(s), processado(s) e salvo(s) na base de conhecimento.")
            for item in ok:
                st.caption(_linha_item(item))
                for aviso in item.get("avisos") or []:
                    st.caption(f"↳ Atenção: {_md(aviso)}")

        if duplicados:
            st.info(f"{len(duplicados)} vídeo(s) ignorado(s) porque já estavam na base:")
            for item in duplicados:
                st.caption(f"{_md(item['arquivo'])} — já cadastrado como **{_md(item['rotulo'])}** ({item['duplicado_de']})")

        if falhas:
            st.error(f"{len(falhas)} vídeo(s) não puderam ser processados (os casos continuam na tela para correção):")
            for item in falhas:
                st.caption(f"{_md(item['arquivo'])}: {_md(item['erro'])}")

        if ok and not resultado.get("interrompido"):
            if resultado.get("erro_treino"):
                st.error(f"Os vídeos foram salvos, mas o treino da IA falhou: {_md(resultado['erro_treino'])}")
            elif resultado["treinou"]:
                texto = f"IA treinada com sucesso com {resultado['n_amostras_treino']} casos!"
                if resultado.get("acuracia_cv") is not None:
                    texto += (
                        f" Acurácia estimada: {resultado['acuracia_cv']:.0%} "
                        f"(validação cruzada em {resultado['n_folds_cv']} partes)."
                    )
                st.success(texto)
                st.caption(f"Classes treinadas: {_md(', '.join(resultado['classes_treinadas']))}")
            else:
                st.info(f"Os dados foram salvos, mas a IA ainda não foi treinada: {_md(resultado['motivo_sem_treino'])}")

        if resultado.get("classes_aguardando"):
            st.caption(
                f"Classes em acumulação (< {MIN_SAMPLES_PER_CLASS} amostras, fora do treino por enquanto): "
                + ", ".join(f"{_md(r)} ({n})" for r, n in resultado["classes_aguardando"].items())
            )
        st.button("Fechar", key="admin_fechar_resultado", on_click=_fechar_resultado_admin)


def _executar_treino_manual() -> None:
    try:
        with st.spinner("Treinando a IA (LightGBM)..."):
            treinou, resultado_treino, motivo = retreinar_agora()
    except PipelineError as exc:
        st.error(f"Não foi possível treinar a IA: {exc}")
        return
    if treinou and resultado_treino is not None:
        texto = f"IA treinada com sucesso com {resultado_treino.n_amostras_treino} casos!"
        if resultado_treino.acuracia_cv is not None:
            texto += f" Acurácia estimada: {resultado_treino.acuracia_cv:.0%}."
        _flash("base", "success", texto, toast="IA treinada com sucesso!")
    else:
        _flash("base", "info", f"A IA não foi treinada: {motivo}")
    st.rerun()


def _render_gestao_casos(status: StatusBase) -> None:
    tabela = status.tabela_casos
    if tabela is None or tabela.empty:
        st.caption("Nenhum caso cadastrado ainda.")
        return
    exibicao = pd.DataFrame({
        "ID": tabela["UUID"],
        "Problema": tabela["rotulo"],
        "Origem": tabela["origem"].replace({"admin": "Alimentação", "diagnostico": "Diagnóstico", "": "—"}),
        "Arquivo enviado": tabela["arquivo_original"].replace({"": "—"}),
        "Cadastrado em": tabela["criado_em"].map(_formatar_data),
    }).iloc[::-1].reset_index(drop=True)  # mais recentes primeiro

    evento = st.dataframe(
        exibicao, hide_index=True, on_select="rerun", selection_mode="multi-row",
        key=f"admin_tabela_casos_{st.session_state['admin_tabela_geracao']}",
    )
    linhas = list(evento.selection.rows) if evento is not None else []
    if not linhas:
        st.caption("Selecione linhas na tabela para excluir casos cadastrados por engano.")
        return

    selecionados = exibicao.iloc[linhas]["ID"].tolist()
    st.caption(
        f"{len(selecionados)} caso(s) selecionado(s). A exclusão apaga o áudio, o laudo e a linha do "
        "dataset; em seguida a IA é retreinada com os casos restantes."
    )
    confirmar = st.checkbox(
        "Confirmo que quero excluir definitivamente os casos selecionados",
        key=f"admin_confirmar_exclusao_{st.session_state['admin_tabela_geracao']}",
    )
    if st.button("Excluir selecionados", disabled=not confirmar, key="admin_excluir_casos"):
        try:
            with st.spinner("Excluindo casos e retreinando a IA..."):
                resumo = excluir_casos(selecionados)
        except PipelineError as exc:
            st.error(f"Não foi possível excluir: {exc}")
            return
        texto = f"{resumo['n_removidos']} caso(s) excluído(s)."
        if resumo["treinou"]:
            texto += " A IA foi retreinada com os casos restantes."
        else:
            texto += f" A IA ficou sem dados suficientes e voltou ao estado inicial: {resumo['motivo_sem_treino']}"
        _flash("base", "success", texto, toast=f"{resumo['n_removidos']} caso(s) excluído(s).")
        st.session_state["admin_tabela_geracao"] += 1
        st.rerun()


def render_painel_base(status: StatusBase, estado_modelo: Optional[EstadoModelo]) -> None:
    """Mostra o que já foi aprendido e quanto falta para cada problema entrar
    no treino -- deixa claro POR QUE a IA ainda não treinou."""
    with st.container(border=True):
        st.markdown("**Base de conhecimento**")
        _mostrar_flash("base")
        if status.erro:
            st.error(_md(status.erro))
            return

        st.caption(
            f"{status.total_casos} caso(s) cadastrado(s) · {len(status.contagens)} problema(s) diferente(s)"
        )
        if estado_modelo is not None:
            st.caption(f"IA {_descricao_modelo(estado_modelo.metadata)}.")
        else:
            st.caption("IA ainda não treinada.")

        for rotulo, n in sorted(status.contagens.items(), key=lambda kv: (-kv[1], kv[0])):
            if n >= MIN_SAMPLES_PER_CLASS:
                texto = f"{_md(rotulo)} — {n} casos · pronto para o treino"
            else:
                texto = f"{_md(rotulo)} — {n}/{MIN_SAMPLES_PER_CLASS} casos · faltam {MIN_SAMPLES_PER_CLASS - n}"
            st.progress(min(n / MIN_SAMPLES_PER_CLASS, 1.0), text=texto)

        if not status.pode_treinar and status.motivo_sem_treino:
            st.caption(f"Para treinar: {_md(status.motivo_sem_treino)}")
        if status.modelo_desatualizado:
            st.info("Há casos cadastrados que ainda não foram incorporados à IA.")
            if st.button("Treinar IA agora", key="admin_treinar_agora", type="primary"):
                _executar_treino_manual()

        with st.expander(f"Gerenciar casos cadastrados ({status.total_casos})"):
            _render_gestao_casos(status)


def render_tab_admin(status: StatusBase, estado_modelo: Optional[EstadoModelo]) -> None:
    if not render_acesso_admin():
        return
    _recuperar_envio_interrompido()

    st.subheader("Alimentação do Sistema (Admin)")
    st.caption(
        "Cadastre um ou mais vídeos conhecidos, cada um com o laudo textual e o nome exato "
        "do problema. Use o botão “+ Novo caso” para alimentar vários problemas de uma só vez; "
        "num mesmo caso você pode enviar vários vídeos do mesmo problema."
    )
    st.info(
        f"Não há limite de tamanho de arquivo, mas se o vídeo passar de "
        f"{MAX_VIDEO_DURATION_WARNING_S // 60} minutos, corte para os melhores momentos e "
        f"mantenha o vídeo dentro desse limite antes de enviar -- vídeos mais curtos e "
        f"focados no defeito geram um treino mais rápido e mais preciso."
    )

    render_painel_base(status, estado_modelo)

    rotulos_existentes = sorted(status.contagens)
    if rotulos_existentes:
        st.caption(
            "Problemas já cadastrados (use o mesmo nome para somar casos ao mesmo problema; "
            "maiúsculas, minúsculas e acentos são unificados automaticamente): "
            + " · ".join(_md(r) for r in rotulos_existentes)
        )

    casos: List[str] = st.session_state["admin_casos"]
    entradas: List[EntradaCaso] = []
    for numero, caso_id in enumerate(casos, start=1):
        with st.container(border=True):
            col_titulo, col_remover = st.columns([5, 1], vertical_alignment="center")
            with col_titulo:
                st.markdown(f"**Caso {numero}**")
            with col_remover:
                if len(casos) > 1:
                    st.button("Remover", key=f"admin_remover_{caso_id}",
                              on_click=_remover_caso_admin, args=(caso_id,))

            erros_caso = st.session_state["admin_erros_casos"].get(caso_id)
            if erros_caso:
                st.error("Este caso não foi cadastrado. Corrija e clique em Alimentar Sistema de novo:\n\n"
                         + "\n\n".join(f"• {_md(e)}" for e in erros_caso))

            arquivos = st.file_uploader(
                "Vídeo(s) conhecido(s) do mesmo problema", type=TIPOS_ACEITOS,
                accept_multiple_files=True, key=f"admin_video_{caso_id}",
                help="Formatos aceitos: " + ", ".join(TIPOS_ACEITOS)
                     + ". Vários vídeos no mesmo caso compartilham o laudo e o rótulo abaixo.",
            )
            texto_laudo = st.text_area(
                "Texto do laudo do respectivo vídeo", key=f"admin_texto_{caso_id}", height=100
            )
            rotulo = st.text_input(
                "Nome exato do problema (rótulo)", key=f"admin_rotulo_{caso_id}"
            )
            modelo_maquina = st.text_input(
                "Modelo da máquina/equipamento", key=f"admin_modelo_{caso_id}",
                help="Ex.: modelo comercial, família ou código interno da máquina.",
            )
            componente_recomendado = st.text_input(
                "Peça ou conjunto prioritário para inspeção", key=f"admin_componente_{caso_id}",
                help="Ex.: rolamento do motor, correia de transmissão, conjunto da bomba.",
            )
            causas_provaveis = st.text_area(
                "Possíveis causas/origens confirmadas", key=f"admin_causas_{caso_id}", height=88,
                help="Registre causas observadas neste tipo de falha: desgaste, desalinhamento, obstrução etc.",
            )
            rotulo_limpo = limpar_texto(rotulo)
            if rotulo_limpo:
                canonico = normalizar_rotulo(rotulo_limpo, rotulos_existentes)
                if canonico != rotulo_limpo:
                    st.caption(f"Será somado ao problema já cadastrado **{_md(canonico)}**.")
                elif canonico not in rotulos_existentes:
                    st.caption("Problema novo — será criado na base.")
            entradas.append(EntradaCaso(
                caso_id=caso_id, numero=numero, arquivos=list(arquivos or []),
                texto=texto_laudo or "", rotulo=rotulo or "", modelo_maquina=modelo_maquina or "",
                componente_recomendado=componente_recomendado or "", causas_provaveis=causas_provaveis or "",
            ))

    st.button("+ Novo caso", key="admin_novo_caso", on_click=_adicionar_caso_admin)

    st.markdown("---")

    if st.button("Alimentar Sistema", type="primary", width="stretch", key="admin_alimentar"):
        preenchidas = [
            e for e in entradas if e.arquivos or limpar_texto(e.texto) or limpar_texto(e.rotulo)
            or limpar_texto(e.modelo_maquina) or limpar_texto(e.componente_recomendado) or limpar_texto(e.causas_provaveis)
        ]
        if not preenchidas:
            st.warning("Preencha ao menos um caso antes de alimentar o sistema.")
        elif problemas := validar_entradas_admin(preenchidas):
            st.warning("Complete os campos antes de alimentar o sistema:\n\n"
                       + "\n\n".join(f"• {p}" for p in problemas))
        else:
            _executar_alimentacao(preenchidas)

    render_resultado_admin()


# ----------------------------------------------------------------- Diagnóstico

def _descartar_diagnostico() -> None:
    st.session_state["diag_resultado"] = None
    st.session_state["diag_geracao"] += 1  # novas chaves => formulário e upload limpos


def _absorver_diagnostico(rotulo_escolhido: str, resultado: Dict[str, Any]) -> None:
    try:
        with st.spinner("Salvando o caso e retreinando a IA..."):
            absorcao = absorver_caso(
                rotulo_escolhido,
                np.array(resultado["features_audio"], dtype=np.float64),
                resultado["texto_laudo"],
                resultado["wav_bytes"],
                hash_video=resultado.get("hash_video", ""),
                nome_arquivo=resultado.get("nome_arquivo", ""),
            )
    except PipelineError as exc:
        st.error(f"Não foi possível absorver o caso: {exc}")
        return
    except Exception as exc:
        logger.exception("Erro inesperado ao absorver caso de diagnóstico")
        st.error(f"Ocorreu um erro inesperado ao treinar a nova amostra: {exc}")
        return

    if absorcao.duplicado_de:
        _flash("diag", "info",
               f"Este vídeo já estava na base como **{_md(absorcao.rotulo)}** ({absorcao.duplicado_de}); "
               "nada foi alterado.")
    elif absorcao.treinou:
        _flash("diag", "success",
               f"Caso registrado como **{_md(absorcao.rotulo)}** ({absorcao.registro_id}). "
               "A IA absorveu este caso e ficou mais inteligente!",
               toast="Caso registrado e IA retreinada!")
    else:
        _flash("diag", "success",
               f"Caso registrado como **{_md(absorcao.rotulo)}** ({absorcao.registro_id}).",
               toast="Caso registrado!")
        _flash("diag", "info", f"A IA ainda não foi (re)treinada: {_md(absorcao.motivo_sem_treino)}")
    if absorcao.classes_aguardando:
        _flash("diag", "caption",
               f"Classes em acumulação (< {MIN_SAMPLES_PER_CLASS} amostras): "
               + ", ".join(f"{_md(r)} ({n})" for r, n in absorcao.classes_aguardando.items()))

    _descartar_diagnostico()
    st.rerun()


def render_tab_diagnostico(estado_modelo: Optional[EstadoModelo], status: StatusBase) -> None:
    st.subheader("Diagnóstico Inteligente (Operador)")
    _mostrar_flash("diag")
    geracao = st.session_state["diag_geracao"]
    resultado: Optional[Dict[str, Any]] = st.session_state.get("diag_resultado")

    if resultado is None:
        if estado_modelo is None:
            st.info(
                "O diagnóstico fica disponível depois do primeiro treino da IA. "
                + (f"Para treinar: {_md(status.motivo_sem_treino)}" if status.motivo_sem_treino else
                   "Use o botão “Treinar IA agora” na aba de Alimentação.")
            )
            return

        with st.form(f"form_diagnostico_{geracao}", clear_on_submit=False):
            arquivo_video = st.file_uploader(
                "Vídeo desconhecido", type=TIPOS_ACEITOS, key=f"upload_diagnostico_{geracao}"
            )
            modelo_maquina = st.text_input(
                "Modelo da máquina/equipamento", key=f"modelo_diagnostico_{geracao}",
                help="O relatório busca referências cadastradas para este modelo.",
            )
            texto_laudo = st.text_area(
                "Texto do laudo do respectivo vídeo", key=f"texto_diagnostico_{geracao}", height=120
            )
            enviado = st.form_submit_button("Gerar Diagnóstico")

        if not enviado:
            return
        if arquivo_video is None:
            st.warning("Envie um vídeo desconhecido antes de gerar o diagnóstico.")
            return
        if not limpar_texto(texto_laudo):
            st.warning("O texto do laudo não pode estar vazio.")
            return
        if not limpar_texto(modelo_maquina):
            st.warning("Informe o modelo da máquina para receber recomendações direcionadas.")
            return

        barra = st.progress(0.0, text="Preparando...")
        try:
            novo = gerar_diagnostico(
                arquivo_video.getvalue(), arquivo_video.name, texto_laudo, modelo_maquina, estado_modelo,
                etapa=lambda texto, fracao: barra.progress(min(fracao, 1.0), text=_md(f"{texto}...")),
            )
        except PipelineError as exc:
            barra.empty()
            st.error(f"Não foi possível gerar o diagnóstico: {exc}")
            return
        except Exception as exc:
            logger.exception("Erro inesperado ao gerar diagnóstico")
            barra.empty()
            st.error(f"Ocorreu um erro inesperado ao gerar o diagnóstico: {exc}")
            return
        st.session_state["diag_resultado"] = novo
        st.rerun()
        return

    # ---- Passo B: resultado dinâmico (persistido em session_state) ----
    probs: Dict[str, float] = resultado["probs"]
    metadados = resultado.get("metadados") or {}
    descricao = f"Arquivo analisado: {_md(resultado.get('nome_arquivo', ''))}"
    if metadados:
        descricao += f" · {metadados['duracao_s']:.1f} s de áudio @ {metadados['sample_rate']} Hz"
    if resultado.get("ratio_fala") is not None and resultado["ratio_fala"] >= MIN_SPEECH_RATIO:
        descricao += (" · voz humana removida" if resultado.get("demucs_aplicado")
                      else " · voz humana detectada (não removida)")
    st.caption(descricao + ".")
    for aviso in resultado.get("avisos") or []:
        st.caption(f"↳ Atenção: {_md(aviso)}")

    render_relatorio_diagnostico(resultado)
    st.caption(
        "O modo operador é somente consulta: a confirmação, o cadastro e o treinamento da base são feitos na área administrativa."
    )
    st.button("Analisar outro vídeo", key=f"diag_descartar_{geracao}", on_click=_descartar_diagnostico)


# -------------------------------------------------------------------- Sidebar

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Backup e migração")
        st.caption(
            "Tudo que a IA aprendeu fica salvo em 4 itens, do lado do app.py: "
            "`dataset_master.csv`, `audios_conhecidos/`, `textos_conhecidos/` e `modelos/`. "
            "Para levar para outro PC, copie esses 4 itens (ou baixe o .zip abaixo) para a "
            "mesma pasta onde está o app.py no outro computador -- não é necessário treinar de novo."
        )
        st.download_button(
            "Baixar backup (.zip)", data=gerar_backup_zip,
            file_name=f"diagnostico_acustico_backup_{datetime.now():%Y-%m-%d}.zip",
            mime="application/zip", width="stretch", on_click="ignore",
        )

        st.markdown("### Verificação do ambiente")
        for nome, ok, detalhe in verificar_ambiente():
            st.caption(f"{'✓' if ok else '✗'} **{nome}** — {_md(detalhe)}")


def configurar_pagina() -> None:
    st.set_page_config(page_title="Diagnóstico Acústico de Falhas Mecânicas", layout="centered")
    st.markdown(
        """
        <style>
        /* ================================================================
           VISUAL SYSTEM — white editorial / automotive minimalism
           Camada exclusivamente visual. Não altera estado, dados ou pipeline.
           ================================================================ */

        @property --edge-angle {
            syntax: "<angle>";
            initial-value: 0deg;
            inherits: false;
        }

        :root {
            --bg: #ffffff;
            --surface: #fafafa;
            --surface-raised: #f5f5f3;
            --text: #111111;
            --muted: #666662;
            --muted-2: #8a8a85;
            --line: #e6e6e2;
            --line-strong: #c9c9c3;
            --focus: #111111;
            --accent: #c8102e;
            --accent-soft: #fff5f7;
            --accent-line: #edc6ce;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--bg) !important;
            color: var(--text) !important;
        }

        html, body, p, div, span, label, input, textarea, button {
            font-family: "Helvetica Neue", "Inter", "Segoe UI", Arial, sans-serif;
        }

        [data-testid="stAppViewContainer"] > .main {
            background: var(--bg) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.94) !important;
            border-bottom: 1px solid rgba(17,17,17,0.06);
            backdrop-filter: blur(14px);
        }

        .main .block-container {
            max-width: 1080px;
            padding-top: 3.6rem;
            padding-bottom: 6rem;
            padding-left: 2.3rem;
            padding-right: 2.3rem;
        }

        /* Header institucional — logos em uma faixa própria acima do título.
           Isso elimina qualquer possibilidade de sobreposição com o texto. */
        h1 {
            position: relative;
            width: 100%;
            max-width: 900px;
            min-height: 150px;
            box-sizing: border-box;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            margin: 0 auto 1rem auto !important;
            padding: 92px 0 1.15rem 0;
            border-bottom: 1px solid var(--line);
            color: var(--text) !important;
            font-family: "Iowan Old Style", Baskerville, "Times New Roman", Georgia, serif !important;
            font-size: clamp(1.72rem, 2.25vw, 2.08rem) !important;
            font-weight: 400 !important;
            line-height: 1.08 !important;
            letter-spacing: -0.026em !important;
            text-align: center;
            text-wrap: balance;
        }

        h1::before,
        h1::after {
            content: "";
            position: absolute;
            top: 8px;
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
            pointer-events: none;
        }

        h1::before {
            left: 1.5%;
            width: 132px;
            height: 64px;
            background-image: url("data:image/png;base64,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");
        }

        h1::after {
            right: 1.5%;
            width: 218px;
            height: 68px;
            background-image: url("data:image/png;base64,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");
        }

        h2, h3 {
            color: var(--text) !important;
            font-family: "Iowan Old Style", Baskerville, "Times New Roman", Georgia, serif !important;
            font-weight: 400 !important;
            letter-spacing: -0.022em !important;
        }

        h2 { font-size: 2.08rem !important; }
        h3 { font-size: 1.52rem !important; }

        [data-testid="stCaptionContainer"],
        .stCaption,
        [data-testid="stMarkdownContainer"] p {
            color: var(--muted) !important;
            line-height: 1.65 !important;
        }

        [data-testid="stMarkdownContainer"] strong {
            color: var(--text) !important;
            font-weight: 600 !important;
        }

        hr {
            border: 0 !important;
            border-top: 1px solid var(--line) !important;
            margin: 2.2rem 0 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #f7f7f5 !important;
            border-right: 1px solid var(--line) !important;
        }

        [data-testid="stSidebar"] > div {
            background: #f7f7f5 !important;
        }

        [data-testid="stSidebar"] h3 {
            font-family: "Helvetica Neue", "Inter", "Segoe UI", Arial, sans-serif !important;
            font-size: 0.72rem !important;
            font-weight: 650 !important;
            line-height: 1.25 !important;
            letter-spacing: 0.15em !important;
            text-transform: uppercase !important;
            color: var(--text) !important;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #72726d !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2.2rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            height: 3.2rem;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            color: #868680 !important;
            font-size: 0.72rem !important;
            font-weight: 650 !important;
            letter-spacing: 0.13em !important;
            text-transform: uppercase !important;
            transition: color 180ms ease !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--accent) !important;
        }

        .stTabs [data-baseweb="tab"] p {
            color: inherit !important;
            font-size: inherit !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--text) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            height: 2px !important;
            background: var(--accent) !important;
        }

        .stTabs [data-baseweb="tab-border"] {
            background: transparent !important;
        }

        /* Cards */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            transition: border-color 180ms ease !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:focus-within {
            border-color: var(--line-strong) !important;
        }

        /* Widget labels */
        [data-testid="stWidgetLabel"] p,
        label[data-testid="stWidgetLabel"] p,
        .stTextInput label p,
        .stTextArea label p,
        [data-testid="stFileUploader"] label p {
            color: #242421 !important;
            font-family: "Helvetica Neue", "Inter", "Segoe UI", Arial, sans-serif !important;
            font-size: 0.70rem !important;
            font-weight: 650 !important;
            letter-spacing: 0.12em !important;
            line-height: 1.3 !important;
            text-transform: uppercase !important;
        }

        /* Inputs */
        .stTextInput input,
        .stTextArea textarea {
            color: var(--text) !important;
            background: #ffffff !important;
            border: 1px solid #d8d8d3 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            caret-color: var(--accent) !important;
            font-size: 0.95rem !important;
            transition: border-color 180ms ease, box-shadow 180ms ease !important;
        }

        .stTextInput input { min-height: 3rem !important; }

        .stTextArea textarea {
            padding: 0.95rem 1rem !important;
            line-height: 1.55 !important;
        }

        .stTextInput input:hover,
        .stTextArea textarea:hover {
            border-color: #b9b9b4 !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px rgba(200,16,46,0.14) !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #9a9a95 !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            min-height: 68px !important;
            padding: 0.75rem !important;
            background: #fbfbfa !important;
            border: 1px solid #d8d8d3 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            transition: border-color 180ms ease, background-color 180ms ease !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #b9b9b4 !important;
            background: #ffffff !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }

        [data-testid="stFileUploaderDropzone"] > svg,
        [data-testid="stFileUploaderDropzone"] button svg,
        [data-testid="stFileUploader"] section > svg {
            display: none !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            min-width: 190px !important;
            min-height: 42px !important;
            margin: 0 !important;
        }

        [data-testid="stFileUploader"] small { display: none !important; }

        /* ================================================================
           BOTÕES — borda vermelha "desenhada" progressivamente no hover.
           O pseudo-elemento usa um conic-gradient mascarado como um anel.
           Ao passar o mouse, --edge-angle percorre 0 -> 360 graus em 800 ms.
           ================================================================ */
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button,
        [data-testid="stFileUploaderDropzone"] button {
            --edge-angle: 0deg;
            position: relative !important;
            isolation: isolate;
            overflow: hidden;
            min-height: 44px !important;
            padding: 0.72rem 1.15rem !important;
            color: var(--text) !important;
            background: #ffffff !important;
            border: 1px solid #cfcfca !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            font-family: "Helvetica Neue", "Inter", "Segoe UI", Arial, sans-serif !important;
            font-size: 0.70rem !important;
            font-weight: 650 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            transition:
                color 220ms ease,
                background-color 220ms ease,
                border-color 220ms ease !important;
        }

        .stButton > button::before,
        .stDownloadButton > button::before,
        .stFormSubmitButton > button::before,
        [data-testid="stFileUploaderDropzone"] button::before {
            content: "";
            position: absolute;
            inset: -1px;
            z-index: 2;
            padding: 2px;
            pointer-events: none;
            background: conic-gradient(
                from -90deg,
                var(--accent) 0deg,
                var(--accent) var(--edge-angle),
                transparent var(--edge-angle),
                transparent 360deg
            );
            -webkit-mask:
                linear-gradient(#000 0 0) content-box,
                linear-gradient(#000 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            transition: --edge-angle 800ms cubic-bezier(0.22, 1, 0.36, 1);
        }

        .stButton > button > *,
        .stDownloadButton > button > *,
        .stFormSubmitButton > button > *,
        [data-testid="stFileUploaderDropzone"] button > * {
            position: relative;
            z-index: 3;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {
            --edge-angle: 360deg;
            color: var(--accent) !important;
            background: #ffffff !important;
            border-color: #d9d9d4 !important;
        }

        .stButton > button:focus-visible,
        .stDownloadButton > button:focus-visible,
        .stFormSubmitButton > button:focus-visible,
        [data-testid="stFileUploaderDropzone"] button:focus-visible {
            outline: 2px solid rgba(200,16,46,0.25) !important;
            outline-offset: 2px !important;
        }

        /* Primary: preto sólido; no hover a borda vermelha percorre o contorno. */
        button[kind="primary"],
        button[data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primary"] {
            color: #ffffff !important;
            background: #111111 !important;
            border-color: #111111 !important;
        }

        button[kind="primary"] p,
        button[data-testid="stBaseButton-primary"] p,
        [data-testid="stBaseButton-primary"] p {
            color: #ffffff !important;
        }

        button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover {
            color: #ffffff !important;
            background: #111111 !important;
            border-color: #111111 !important;
        }

        /* Alerts / mensagens — branco + carmesim, sem caixas coloridas pesadas. */
        [data-testid="stAlert"],
        [data-baseweb="notification"] {
            color: var(--text) !important;
            background: var(--accent-soft) !important;
            border: 1px solid var(--accent-line) !important;
            border-left: 3px solid var(--accent) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }

        [data-testid="stAlert"] > div,
        [data-baseweb="notification"] > div {
            background: transparent !important;
        }

        [data-testid="stAlert"] svg,
        [data-baseweb="notification"] svg {
            display: none !important;
        }

        [data-testid="stAlert"] p,
        [data-baseweb="notification"] p {
            color: #2f2f2b !important;
            font-size: 0.91rem !important;
            line-height: 1.65 !important;
        }

        [data-testid="stAlert"] a { color: var(--accent) !important; }

        /* Spinner */
        [data-testid="stSpinner"] > div,
        [data-testid="stSpinner"] svg {
            color: var(--accent) !important;
        }

        /* Uploaded-file rows */
        [data-testid="stFileUploaderFile"] {
            background: #ffffff !important;
            border-top: 1px solid var(--line) !important;
            border-radius: 0 !important;
        }

        ::selection {
            background: rgba(200,16,46,0.16);
            color: #111111;
        }

        a {
            color: var(--text) !important;
            text-underline-offset: 0.18em;
        }

        a:hover { color: var(--accent) !important; }

        @media (prefers-reduced-motion: reduce) {
            .stButton > button::before,
            .stDownloadButton > button::before,
            .stFormSubmitButton > button::before,
            [data-testid="stFileUploaderDropzone"] button::before {
                transition-duration: 0.01ms !important;
            }
        }

        /* Responsive */
        @media (max-width: 760px) {
            .main .block-container {
                padding-top: 2.6rem;
                padding-left: 1.15rem;
                padding-right: 1.15rem;
            }

            h1 {
                min-height: 142px;
                padding-top: 84px;
                font-size: clamp(1.5rem, 6.2vw, 1.92rem) !important;
                line-height: 1.10 !important;
            }

            h1::before {
                left: 0;
                width: 92px;
                height: 54px;
            }

            h1::after {
                right: 0;
                width: 142px;
                height: 56px;
            }

            .stTabs [data-baseweb="tab-list"] { gap: 1.1rem; }

            [data-testid="stFileUploaderDropzone"] button {
                width: 100% !important;
                min-width: 0 !important;
            }
        }

        /* ================================================================
           COMPLEMENTOS — feedback de envio, progresso e painéis.
           Mesma linguagem visual: branco, linhas finas, carmesim de acento.
           ================================================================ */

        /* Tipos de mensagem: o carmesim fica reservado para erros e avisos;
           confirmações usam preto e informações neutras usam cinza. */
        [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
            background: var(--surface-raised) !important;
            border-color: var(--line) !important;
            border-left-color: var(--text) !important;
        }

        [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            border-left-color: var(--muted-2) !important;
        }

        /* Notificação flutuante (toast) */
        [data-testid="stToast"] {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--line) !important;
            border-left: 3px solid var(--text) !important;
            border-radius: 0 !important;
            box-shadow: 0 10px 30px rgba(17, 17, 17, 0.08) !important;
        }

        /* Barra de progresso: fina, reta, carmesim */
        [data-testid="stProgress"] [role="progressbar"],
        [data-testid="stProgress"] [role="progressbar"] > div,
        [data-testid="stProgress"] [role="progressbar"] div {
            border-radius: 0 !important;
        }

        [data-testid="stProgress"] p {
            color: var(--muted) !important;
            font-size: 0.82rem !important;
        }

        /* Expander (gerenciar casos) */
        [data-testid="stExpander"] details {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            border-radius: 0 !important;
        }

        [data-testid="stExpander"] summary p {
            color: #242421 !important;
            font-size: 0.70rem !important;
            font-weight: 650 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
        }

        [data-testid="stExpander"] summary:hover p {
            color: var(--accent) !important;
        }

        /* Tabela de casos */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line) !important;
            border-radius: 0 !important;
        }

        /* Checkbox de confirmação: frase longa, então sem a caixa alta dos rótulos */
        [data-testid="stCheckbox"] label p,
        [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p {
            color: #2f2f2b !important;
            font-size: 0.88rem !important;
            font-weight: 400 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
        }

        /* Botões desabilitados */
        .stButton > button:disabled,
        .stButton > button:disabled:hover {
            color: #a5a5a0 !important;
            background: #f7f7f5 !important;
            border-color: var(--line) !important;
            --edge-angle: 0deg;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configurar_pagina()
    inicializar_sessao()

    st.title("Diagnóstico Acústico de Falhas Mecânicas")
    st.caption("Análise multimodal (áudio + texto) para diagnóstico de falhas em equipamentos.")

    estado_modelo = carregar_modelo_salvo()
    status = obter_status_base(estado_modelo)

    if status.erro:
        st.error(f"Problema na base de conhecimento: {_md(status.erro)}")
    if estado_modelo is None:
        st.warning(
            "**Cold start** — ainda não existe um modelo treinado "
            f"(`{MODEL_PATH.relative_to(BASE_DIR).as_posix()}`). Use a aba **Alimentação (Admin)** para "
            f"cadastrar os primeiros casos conhecidos. Cada rótulo precisa acumular pelo menos "
            f"{MIN_SAMPLES_PER_CLASS} amostras para entrar no treino, e o primeiro treino exige "
            f"{MIN_CLASSES_TREINO} rótulos diferentes."
        )
    else:
        classes = estado_modelo.metadata.get("classes", [])
        st.caption(
            f"Modelo carregado — {len(classes)} classe(s) conhecida(s): {_md(', '.join(classes))}."
        )

    aba_admin, aba_diagnostico = st.tabs(["Alimentação (Admin)", "Diagnóstico (Operador)"])
    with aba_admin:
        render_tab_admin(status, estado_modelo)
    with aba_diagnostico:
        render_tab_diagnostico(estado_modelo, status)

    render_sidebar()  # por último: a checagem do ambiente não atrasa o conteúdo principal


if __name__ == "__main__":
    main()
