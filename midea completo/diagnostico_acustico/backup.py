"""Backup da base de conhecimento e verificação do ambiente."""
from __future__ import annotations

import importlib.util
import io
import zipfile
from typing import List, Tuple

from .audio import localizar_ffmpeg
from .config import (
    AUDIOS_DIR,
    BASE_CONHECIMENTO_PATH,
    BASE_DIR,
    CAIXA_ENTRADA_DIR,
    DATASET_PATH,
    MODELOS_DIR,
    SPACY_MODEL_NAME,
    TEXTOS_DIR,
)
from .erros import PipelineError
from .recursos import carregar_spacy, carregar_vad


# ==============================================================================
# 11. BACKUP / MIGRAÇÃO ENTRE MÁQUINAS
# ==============================================================================

def gerar_backup_zip() -> bytes:
    """Empacota tudo que a IA já aprendeu -- dataset_master.csv, os áudios e
    textos conhecidos, o modelo LightGBM treinado, os textos de cada defeito e
    os casos dos celulares ainda na caixa de entrada -- em um único .zip. Basta
    baixar esse arquivo e extraí-lo na pasta do app.py em outro computador
    para continuar exatamente de onde parou, sem retreinar do zero.

    Só é chamado quando o usuário clica no botão de download (antes, o .zip
    inteiro era montado a cada clique em qualquer lugar do app)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for arquivo in (DATASET_PATH, BASE_CONHECIMENTO_PATH):
            if arquivo.exists():
                zf.write(arquivo, arcname=arquivo.name)
        for pasta in (AUDIOS_DIR, TEXTOS_DIR, MODELOS_DIR, CAIXA_ENTRADA_DIR):
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
