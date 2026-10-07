"""Interface: utilitários e estado da sessão."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st


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
        # Aba Diagnóstico
        "diag_resultado": None,
        "diag_geracao": 0,
        # Modo administrador (vale só para esta aba do navegador; some ao recarregar)
        "admin_autenticado": False,
        # Mensagens exibidas uma única vez depois de um st.rerun()
        "flash": {},
    }
    for chave, valor in padroes.items():
        st.session_state.setdefault(chave, valor)


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
