"""Modelos carregados uma única vez (spaCy, Silero VAD, Demucs) e trava de processamento."""
from __future__ import annotations

import contextlib
import io
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Tuple

import spacy
import streamlit as st
import torch

from .config import DEMUCS_MODEL_NAME, SPACY_MODEL_NAME
from .erros import ModeloIndisponivelError


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
    _silero_pkg = None
    _silero_get_speech_timestamps = None


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
