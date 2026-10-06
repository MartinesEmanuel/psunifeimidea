"""Interface: barra lateral."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from typing import Optional

from .backup import gerar_backup_zip, verificar_ambiente
from .treino import EstadoModelo
from .ui_central import render_sincronizacao_sidebar
from .ui_comum import _md


# -------------------------------------------------------------------- Sidebar

def render_sidebar(estado_modelo: Optional[EstadoModelo] = None) -> None:
    with st.sidebar:
        render_sincronizacao_sidebar(estado_modelo)
        st.markdown("### Backup e migração")
        st.caption(
            "Tudo que a IA aprendeu fica salvo do lado do app.py: "
            "`dataset_master.csv`, `base_conhecimento.json`, `audios_conhecidos/`, `textos_conhecidos/`, "
            "`modelos/` e `caixa_de_entrada/`. Para levar para outro PC, baixe o .zip abaixo e extraia na "
            "mesma pasta onde está o app.py no outro computador -- não é necessário treinar de novo."
        )
        if st.session_state.get("admin_autenticado"):
            st.download_button(
                "Baixar backup (.zip)", data=gerar_backup_zip,
                file_name=f"diagnostico_acustico_backup_{datetime.now():%Y-%m-%d}.zip",
                mime="application/zip", width="stretch", on_click="ignore",
            )
        else:
            st.caption("O botão de backup aparece depois de entrar no modo administrador (aba Alimentação).")

        st.markdown("### Verificação do ambiente")
        for nome, ok, detalhe in verificar_ambiente():
            st.caption(f"{'✓' if ok else '✗'} **{nome}** — {_md(detalhe)}")
