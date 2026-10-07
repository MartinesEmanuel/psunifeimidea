"""Exceções do domínio."""
from __future__ import annotations


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
