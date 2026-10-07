"""Interface: barra lateral."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from typing import Optional

from .backup import gerar_backup_zip, restaurar_backup_zip, verificar_ambiente
from .dataset import nome_exibicao
from .erros import PipelineError
from .pipeline import retreinar_agora
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
            "`modelos/` e `caixa_de_entrada/`. Para levar para outro PC, baixe o .zip abaixo e restaure no "
            "outro computador (ou extraia na mesma pasta onde está o app.py)."
        )
        if st.session_state.get("admin_autenticado"):
            st.download_button(
                "Baixar backup (.zip)", data=gerar_backup_zip,
                file_name=f"diagnostico_acustico_backup_{datetime.now():%Y-%m-%d}.zip",
                mime="application/zip", width="stretch", on_click="ignore",
            )
            _render_restaurar_backup()
        else:
            st.caption("O botão de backup aparece depois de entrar no modo administrador (aba Alimentação).")

        st.markdown("### Verificação do ambiente")
        for nome, ok, detalhe in verificar_ambiente():
            st.caption(f"{'✓' if ok else '✗'} **{nome}** — {_md(detalhe)}")


def _render_restaurar_backup() -> None:
    aviso = st.session_state.pop("backup_restaurado", None)
    if aviso:
        st.success(aviso)
    with st.expander("Restaurar um backup (.zip)"):
        st.caption(
            "Traz a base de um backup -- desta versão do site ou da versão antiga. Os casos, áudios e a IA "
            "de agora são trocados pelos do backup; antes disso, uma cópia do que existe hoje é guardada "
            "em `_versoes_anteriores/`. No fim, a IA é treinada de novo."
        )
        arquivo = st.file_uploader("Arquivo de backup", type=["zip"], key="backup_restaurar_arquivo")
        if arquivo is None or not st.button("Restaurar este backup", key="backup_restaurar", width="stretch"):
            return
        try:
            with st.spinner("Restaurando o backup..."):
                resumo = restaurar_backup_zip(arquivo.getvalue())
        except PipelineError as exc:
            st.error(f"O backup não foi restaurado: {exc}")
            return
        partes = [f"Backup restaurado: {resumo['casos']} caso(s) em {len(resumo['categorias'])} categoria(s) — "
                  + ", ".join(f"{_md(nome_exibicao(c))} ({n})" for c, n in resumo["categorias"].items()) + "."]
        if resumo["copia_automatica"] is not None:
            partes.append(f"O que havia antes foi guardado em `_versoes_anteriores/{resumo['copia_automatica'].name}`.")
        try:
            with st.spinner("Treinando a IA com a base restaurada..."):
                treinou, resultado, motivo = retreinar_agora()
            partes.append("IA treinada de novo." if treinou and resultado is not None
                          else f"A IA ainda não foi treinada: {_md(motivo or '')}")
        except PipelineError as exc:
            partes.append(f"Não foi possível treinar a IA agora ({_md(str(exc))}); use \"Treinar IA agora\" na aba Alimentação.")
        partes.append("Para os celulares receberem, gere um pacote novo em \"Celulares e sincronização\".")
        st.session_state["backup_restaurado"] = " ".join(partes)
        st.rerun()
