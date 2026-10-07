"""Interface: visual do site.

As cores, fontes e logos ficam na pasta estilo/ (ao lado do app.py), que é
compartilhada com o aplicativo do celular: estilo/tokens.json define as cores e
as fontes, estilo/site.css o visual do site e os .png os logos do cabeçalho.
"""
from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import streamlit as st

from .config import BASE_DIR

ESTILO_DIR = BASE_DIR / "estilo"

# Nome de cada cor do tokens.json -> variável CSS usada no site.css.
_VARIAVEIS_CSS = {
    "fundo": "bg", "superficie": "surface", "superficie_elevada": "surface-raised",
    "texto": "text", "texto_suave": "muted", "texto_suave_2": "muted-2",
    "linha": "line", "linha_forte": "line-strong", "foco": "focus",
    "destaque": "accent", "destaque_suave": "accent-soft", "destaque_linha": "accent-line",
}


def _data_uri_png(arquivo: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(arquivo.read_bytes()).decode("ascii")


@lru_cache(maxsize=4)
def _css_montado(_assinatura: Tuple[int, ...]) -> str:
    """Junta as variáveis do tokens.json com o site.css. Só refaz quando algum
    dos arquivos muda (a assinatura são as datas de modificação)."""
    tokens = json.loads((ESTILO_DIR / "tokens.json").read_text(encoding="utf-8"))
    variaveis = [f"--{css}: {tokens['cores'][nome]};" for nome, css in _VARIAVEIS_CSS.items()]
    variaveis += [
        f"--fonte-titulo: {tokens['fontes']['titulo']};",
        f"--fonte-texto: {tokens['fontes']['texto']};",
        f'--logo-esquerda: url("{_data_uri_png(ESTILO_DIR / tokens["logos"]["esquerda"])}");',
        f'--logo-direita: url("{_data_uri_png(ESTILO_DIR / tokens["logos"]["direita"])}");',
    ]
    raiz = ":root {\n" + "".join(f"    {v}\n" for v in variaveis) + "}\n"
    return raiz + (ESTILO_DIR / "site.css").read_text(encoding="utf-8")


def configurar_pagina() -> None:
    st.set_page_config(page_title="Diagnóstico Acústico de Falhas Mecânicas", layout="centered")
    arquivos = sorted(ESTILO_DIR.glob("*"))
    assinatura = tuple(a.stat().st_mtime_ns for a in arquivos)
    st.markdown(f"<style>\n{_css_montado(assinatura)}</style>", unsafe_allow_html=True)
