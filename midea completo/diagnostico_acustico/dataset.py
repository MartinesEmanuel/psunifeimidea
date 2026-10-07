"""Base de casos (dataset_master.csv): rótulos, modelos de máquina e peças."""
from __future__ import annotations

import os
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (
    APELIDOS_REFERENCIA,
    AUDIOS_DIR,
    BASE_DIR,
    DATASET_PATH,
    ROTULO_REFERENCIA,
    TEXTOS_DIR,
    logger,
)
from .erros import DatasetError, ExtracaoAudioError
from .features import nomes_colunas_features, validar_dimensao_vetor
from .texto import limpar_texto


# ==============================================================================
# 8. PERSISTÊNCIA DO DATASET MESTRE (dataset_master.csv)
# ==============================================================================

COLUNAS_META: List[str] = ["UUID", "rotulo", "audio_path", "text_path"]
# Metadados opcionais (datasets antigos, sem essas colunas, continuam funcionando):
#   hash_video       -> SHA-256 do arquivo enviado, usado para barrar duplicatas
#   arquivo_original -> nome do arquivo enviado na interface
#   origem           -> "admin" (aba Alimentação) ou "diagnostico" (confirmação do operador)
#   criado_em        -> data/hora do cadastro
#   modelo_maquina   -> modelo da máquina gravada, digitado por quem cadastra
#   peca             -> peça/componente indicado para o problema (opcional, aba Alimentação)
COLUNAS_META_EXTRAS: List[str] = ["hash_video", "arquivo_original", "origem", "criado_em", "modelo_maquina", "peca"]


def gerar_uuid_curto(df_existente: Optional[pd.DataFrame] = None) -> str:
    """Gera um identificador curto e único no padrão do briefing (ex.: 'ID-8f92a01c3d').
    São 10 caracteres (antes eram 5) porque a base é compartilhada entre vários
    computadores: com 5, dois computadores logo gerariam o mesmo código."""
    existentes = set(df_existente["UUID"]) if df_existente is not None and "UUID" in df_existente.columns else set()
    for _ in range(10):
        candidato = f"ID-{uuid.uuid4().hex[:10]}"
        if candidato not in existentes:
            return candidato
    return f"ID-{uuid.uuid4().hex[:16]}"  # fallback extremamente improvável de colisão


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


def _normalizar_nome(nome: str, existentes: Sequence[str]) -> str:
    """Se o nome digitado equivale a um já cadastrado (ignorando maiúsculas,
    acentos e espaços extras), reaproveita a grafia existente."""
    limpo = " ".join((nome or "").split())
    chave = _chave_rotulo(limpo)
    for existente in existentes:
        if _chave_rotulo(existente) == chave:
            return existente
    return limpo


def eh_referencia(rotulo: str) -> bool:
    """True para a categoria Referência (máquina funcionando normal) e seus apelidos."""
    return _chave_rotulo(rotulo) in APELIDOS_REFERENCIA


def normalizar_rotulo(rotulo: str, rotulos_existentes: Sequence[str]) -> str:
    """Evita que "Rolamento", "rolamento " e "ROLAMENTO" virem três classes
    diferentes: se o rótulo digitado equivale a um já cadastrado, reaproveita
    a grafia existente. "Normal", "sem defeito" etc. viram a Referência."""
    if eh_referencia(rotulo):
        return ROTULO_REFERENCIA
    return _normalizar_nome(rotulo, rotulos_existentes)


def _chave_modelo(modelo: str) -> str:
    """Compara modelos só por letras e números: "EB-0123", "eb 0123" e "EB0123" são o mesmo."""
    return "".join(c for c in _chave_rotulo(modelo) if c.isalnum())


def normalizar_modelo(modelo: str, modelos_existentes: Sequence[str]) -> str:
    """Se o modelo digitado equivale a um já cadastrado, reaproveita a grafia existente."""
    limpo = " ".join((modelo or "").split())
    chave = _chave_modelo(limpo)
    for existente in modelos_existentes:
        if chave and _chave_modelo(existente) == chave:
            return existente
    return limpo


def nome_exibicao(rotulo: str) -> str:
    """Nome mostrado na tela: deixa claro o que é a categoria Referência."""
    return f"{rotulo} (funcionamento normal)" if eh_referencia(rotulo) else rotulo


def rotulos_do_dataset(df: pd.DataFrame) -> List[str]:
    if df.empty or "rotulo" not in df.columns:
        return []
    return sorted({str(r) for r in df["rotulo"] if str(r).strip()})


def modelos_do_dataset(df: pd.DataFrame) -> List[str]:
    if df.empty or "modelo_maquina" not in df.columns:
        return []
    return sorted({str(m) for m in df["modelo_maquina"] if str(m).strip()})


def pecas_registradas(df: pd.DataFrame, rotulo: str, modelo: str) -> Dict[str, List[Tuple[str, int]]]:
    """Peças cadastradas na Alimentação para um problema, separando as do mesmo
    modelo de máquina das de outros modelos (mais citadas primeiro)."""
    resultado: Dict[str, List[Tuple[str, int]]] = {"mesmo_modelo": [], "outros_modelos": []}
    if df.empty or "peca" not in df.columns:
        return resultado
    do_problema = df[(df["rotulo"] == rotulo) & (df["peca"].str.strip() != "")]
    if do_problema.empty:
        return resultado
    mesmo = do_problema["modelo_maquina"].map(_chave_modelo) == _chave_modelo(modelo)
    for chave, parte in (("mesmo_modelo", do_problema[mesmo]), ("outros_modelos", do_problema[~mesmo])):
        grafias: Dict[str, List[str]] = {}  # "Rolamento 6203" e "rolamento 6203" contam juntos
        for peca in parte["peca"]:
            limpo = limpar_texto(peca)
            grafias.setdefault(_chave_rotulo(limpo), []).append(limpo)
        contagem = [(max(g, key=g.count), len(g)) for g in grafias.values()]  # empate: a grafia mais antiga
        resultado[chave] = sorted(contagem, key=lambda kv: (-kv[1], kv[0]))
    return resultado


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
    # Bases de versões antigas: "Normal", "sem defeito"... viram a categoria Referência,
    # para a máquina funcionando normal ser uma classe só.
    df["rotulo"] = [ROTULO_REFERENCIA if eh_referencia(r) else r for r in df["rotulo"]]
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
