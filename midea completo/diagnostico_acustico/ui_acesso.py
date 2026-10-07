"""Interface: entrada no modo administrador."""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st

from .config import SENHA_ADMIN_MIN_CARACTERES, logger
from .senha import conferir_senha_admin, definir_senha_admin, senha_admin_definida


# ------------------------------------------------------ Acesso do administrador

def _validar_nova_senha(nova: str, repetida: str) -> Optional[str]:
    if len(nova or "") < SENHA_ADMIN_MIN_CARACTERES:
        return f"A senha precisa ter pelo menos {SENHA_ADMIN_MIN_CARACTERES} caracteres."
    if nova != repetida:
        return "As duas senhas digitadas não são iguais."
    return None


def _gravar_senha(nova: str) -> bool:
    try:
        definir_senha_admin(nova)
        return True
    except Exception as exc:
        logger.exception("Falha ao gravar a senha do administrador")
        st.error(f"Não foi possível gravar a senha: {exc}")
        return False


def _sair_admin() -> None:
    st.session_state["admin_autenticado"] = False


def render_acesso_admin() -> bool:
    """Porta de entrada da aba Admin. Na primeira vez pede para criar a senha;
    depois pede a senha a cada nova sessão do navegador. Devolve True quando o
    modo administrador está ativo."""
    if st.session_state.get("admin_autenticado"):
        col_txt, col_btn = st.columns([5, 1], vertical_alignment="center")
        with col_txt:
            st.caption("Modo administrador ativo.")
        with col_btn:
            st.button("Sair", key="admin_sair", on_click=_sair_admin, width="stretch")
        with st.expander("Trocar senha do administrador"):
            with st.form("form_trocar_senha", clear_on_submit=True):
                atual = st.text_input("Senha atual", type="password")
                nova = st.text_input("Nova senha", type="password")
                repetida = st.text_input("Repita a nova senha", type="password")
                if st.form_submit_button("Salvar nova senha"):
                    if not conferir_senha_admin(atual):
                        st.error("A senha atual está incorreta.")
                    elif erro := _validar_nova_senha(nova, repetida):
                        st.error(erro)
                    elif _gravar_senha(nova):
                        st.success("Senha alterada.")
        return True

    st.subheader("Acesso do administrador")
    if not senha_admin_definida():
        st.info(
            "Primeiro acesso: crie a senha do administrador. Só quem tiver essa senha poderá "
            "alimentar o sistema, confirmar diagnósticos, excluir casos e baixar o backup. "
            "Os técnicos usam apenas a aba Diagnóstico."
        )
        with st.form("form_criar_senha", clear_on_submit=True):
            nova = st.text_input("Nova senha", type="password")
            repetida = st.text_input("Repita a senha", type="password")
            if st.form_submit_button("Criar senha e entrar", type="primary"):
                if erro := _validar_nova_senha(nova, repetida):
                    st.error(erro)
                elif _gravar_senha(nova):
                    st.session_state["admin_autenticado"] = True
                    st.session_state["sincronizar_ao_entrar"] = True
                    st.rerun()
        return False

    st.caption("Área restrita ao administrador. Os técnicos usam apenas a aba Diagnóstico.")
    with st.form("form_login_admin", clear_on_submit=True):
        senha = st.text_input("Senha do administrador", type="password")
        if st.form_submit_button("Entrar", type="primary"):
            if conferir_senha_admin(senha):
                st.session_state["admin_autenticado"] = True
                st.session_state["sincronizar_ao_entrar"] = True
                st.rerun()
            else:
                time.sleep(1.0)  # atrasa tentativas seguidas de adivinhar a senha
                st.error("Senha incorreta.")
    return False
