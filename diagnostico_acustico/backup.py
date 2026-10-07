"""Backup da base de conhecimento e verificação do ambiente."""
from __future__ import annotations

import importlib.util
import io
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .audio import localizar_ffmpeg
from .config import (
    AUDIOS_DIR,
    BASE_CONHECIMENTO_PATH,
    BASE_DIR,
    CAIXA_ENTRADA_DIR,
    DATASET_PATH,
    MODELOS_DIR,
    ROTULO_REFERENCIA,
    SPACY_MODEL_NAME,
    TEXTOS_DIR,
)
from .dataset import eh_referencia
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


# A planilha, os áudios, os textos e a IA treinada andam juntos: são trocados de uma vez.
# Os textos dos defeitos só são trocados se o backup os trouxer (backups antigos não têm),
# e os casos da caixa de entrada são somados aos daqui, nunca apagados.
_TROCADOS_JUNTOS = (DATASET_PATH, AUDIOS_DIR, TEXTOS_DIR, MODELOS_DIR)
_ARQUIVOS_DO_BACKUP = {DATASET_PATH.name, BASE_CONHECIMENTO_PATH.name}
_PASTAS_DO_BACKUP = {AUDIOS_DIR.name, TEXTOS_DIR.name, MODELOS_DIR.name, CAIXA_ENTRADA_DIR.name}
PASTA_COPIAS_AUTOMATICAS = BASE_DIR / "_versoes_anteriores"


def _destino_no_backup(nome: str) -> Optional[PurePosixPath]:
    """Caminho (relativo à pasta do app) de um item do .zip, ou None se ele não faz
    parte de um backup -- nada de caminhos absolutos, com ".." ou fora das pastas da base."""
    caminho = PurePosixPath(nome.replace("\\", "/"))
    partes = caminho.parts
    if not partes or caminho.is_absolute() or ".." in partes or ":" in partes[0]:
        return None
    if len(partes) == 1 and partes[0] in _ARQUIVOS_DO_BACKUP:
        return caminho
    if len(partes) > 1 and partes[0] in _PASTAS_DO_BACKUP:
        return caminho
    return None


def _tem_base_atual() -> bool:
    if BASE_CONHECIMENTO_PATH.exists():
        return True
    if DATASET_PATH.exists() and len(DATASET_PATH.read_bytes().splitlines()) > 1:
        return True
    return any(p.is_file() for pasta in (AUDIOS_DIR, TEXTOS_DIR, MODELOS_DIR, CAIXA_ENTRADA_DIR)
               if pasta.exists() for p in pasta.rglob("*"))


def restaurar_backup_zip(dados: bytes) -> Dict[str, Any]:
    """Coloca no lugar a base de um backup (.zip) desta versão ou das anteriores do site.
    Antes, o que existe hoje é guardado em _versoes_anteriores/ (cópia automática).
    Depois disso, a IA deve ser treinada de novo (a interface faz isso sozinha)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(dados))
    except zipfile.BadZipFile as exc:
        raise PipelineError("o arquivo enviado não é um .zip válido.") from exc
    with zf:
        itens = [(info, _destino_no_backup(info.filename)) for info in zf.infolist() if not info.is_dir()]
        validos = [(info, destino) for info, destino in itens if destino is not None]
        if PurePosixPath(DATASET_PATH.name) not in {destino for _, destino in validos}:
            raise PipelineError("este .zip não é um backup do site (não tem o dataset_master.csv).")
        try:
            planilha = pd.read_csv(io.BytesIO(zf.read(DATASET_PATH.name)), usecols=lambda c: c == "rotulo", dtype=str)
        except Exception as exc:
            raise PipelineError(f"o dataset_master.csv do backup está ilegível ({exc}).") from exc
        if "rotulo" not in planilha.columns or planilha.empty:
            raise PipelineError("o backup não tem nenhum caso cadastrado; nada foi alterado.")

        copia = None
        if _tem_base_atual():
            PASTA_COPIAS_AUTOMATICAS.mkdir(exist_ok=True)
            copia = PASTA_COPIAS_AUTOMATICAS / f"backup_automatico_antes_de_restaurar_{datetime.now():%Y-%m-%d_%H%M%S}.zip"
            copia.write_bytes(gerar_backup_zip())

        temporaria = BASE_DIR / f".restaurando_{uuid.uuid4().hex[:8]}"
        try:
            for info, destino in validos:
                alvo = temporaria.joinpath(*destino.parts)
                alvo.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as origem, open(alvo, "wb") as saida:
                    shutil.copyfileobj(origem, saida)
            trocar = list(_TROCADOS_JUNTOS)
            if (temporaria / BASE_CONHECIMENTO_PATH.name).exists():
                trocar.append(BASE_CONHECIMENTO_PATH)
            try:
                for caminho in trocar:
                    if caminho.is_dir():
                        shutil.rmtree(caminho)
                    elif caminho.exists():
                        caminho.unlink()
                    novo = temporaria / caminho.name
                    if novo.exists():
                        shutil.move(str(novo), str(caminho))
                casos_recebidos = temporaria / CAIXA_ENTRADA_DIR.name
                if casos_recebidos.exists():
                    CAIXA_ENTRADA_DIR.mkdir(exist_ok=True)
                    for pasta_caso in casos_recebidos.iterdir():
                        if not (CAIXA_ENTRADA_DIR / pasta_caso.name).exists():
                            shutil.move(str(pasta_caso), str(CAIXA_ENTRADA_DIR / pasta_caso.name))
            except OSError as exc:
                guardado = f" O que havia antes está guardado em _versoes_anteriores/{copia.name}." if copia else ""
                raise PipelineError(
                    f"a restauração parou no meio ({exc}). Feche programas que estejam usando a pasta "
                    f"(ou pause o OneDrive) e tente de novo.{guardado}"
                ) from exc
        finally:
            shutil.rmtree(temporaria, ignore_errors=True)

    rotulos = [ROTULO_REFERENCIA if eh_referencia(str(r)) else str(r) for r in planilha["rotulo"].fillna("")]
    contagem = pd.Series([r for r in rotulos if r.strip()]).value_counts()
    return {
        "casos": int(contagem.sum()),
        "categorias": {str(k): int(v) for k, v in contagem.items()},
        "copia_automatica": copia,
        "ignorados": len(itens) - len(validos),
    }


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
