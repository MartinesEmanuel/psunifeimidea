"""Textos de cada defeito: possíveis causas, o que verificar e o que fazer.

O administrador escreve os textos no site; eles ficam guardados em
base_conhecimento.json (que entra no backup) e vão para os celulares junto com
a IA, para o técnico ver na hora, mesmo sem internet.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Optional

from .config import BASE_CONHECIMENTO_PATH, logger
from .dataset import _chave_rotulo, _gravar_atomico
from .erros import DatasetError
from .texto import limpar_texto

# Campos de cada defeito, na ordem em que aparecem na tela e no relatório.
CAMPOS_TEXTO: Dict[str, str] = {
    "causas": "Possíveis causas",
    "verificar": "O que verificar",
    "o_que_fazer": "O que fazer",
}


def carregar_textos() -> Dict[str, Dict[str, str]]:
    """categoria -> {causas, verificar, o_que_fazer, atualizado_em}."""
    if not BASE_CONHECIMENTO_PATH.exists():
        return {}
    try:
        dados = json.loads(BASE_CONHECIMENTO_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetError(f"base_conhecimento.json corrompido ou ilegível: {exc}") from exc
    return dict(dados.get("categorias", {}))


def textos_da_categoria(categoria: str) -> Optional[Dict[str, str]]:
    """Textos de uma categoria (sem diferenciar maiúsculas e acentos), ou None
    se ainda não foram escritos."""
    try:
        textos = carregar_textos()
    except DatasetError:
        logger.exception("Falha ao ler os textos dos defeitos")
        return None
    chave = _chave_rotulo(categoria)
    for nome, campos in textos.items():
        if _chave_rotulo(nome) == chave and any(str(campos.get(c, "")).strip() for c in CAMPOS_TEXTO):
            return campos
    return None


def salvar_textos_categoria(categoria: str, campos: Dict[str, str], atualizado_em: Optional[str] = None) -> None:
    """Grava os textos de uma categoria. Tudo vazio = textos apagados: a categoria
    fica registrada vazia (com a data), para a exclusão chegar aos outros
    computadores na base compartilhada. [atualizado_em] vem de outro computador."""
    textos = carregar_textos()
    textos = {nome: c for nome, c in textos.items() if _chave_rotulo(nome) != _chave_rotulo(categoria)}
    limpos = {c: str(campos.get(c, "")).strip() for c in CAMPOS_TEXTO}
    textos[limpar_texto(categoria)] = {**limpos, "atualizado_em": atualizado_em or datetime.now().isoformat(timespec="seconds")}
    dados = {"formato": 1, "categorias": dict(sorted(textos.items()))}
    try:
        _gravar_atomico(
            BASE_CONHECIMENTO_PATH,
            lambda caminho: caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8"),
        )
    except Exception as exc:
        raise DatasetError(f"não foi possível salvar os textos dos defeitos: {exc}") from exc
