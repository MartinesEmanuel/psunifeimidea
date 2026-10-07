"""Interface: aba Diagnóstico."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from .config import (
    MIN_SAMPLES_PER_CLASS,
    MIN_SPEECH_RATIO,
    ROTULO_REFERENCIA,
    TIPOS_ACEITOS,
    logger,
)
from .conhecimento import CAMPOS_TEXTO, textos_da_categoria
from .dataset import eh_referencia, nome_exibicao
from .erros import PipelineError
from .pipeline import StatusBase, absorver_caso, gerar_diagnostico
from .texto import limpar_texto
from .treino import EstadoModelo
from .ui_comum import _flash, _md, _mostrar_flash


# ----------------------------------------------------------------- Diagnóstico

def _descartar_diagnostico() -> None:
    st.session_state["diag_resultado"] = None
    st.session_state["diag_geracao"] += 1  # novas chaves => formulário e upload limpos


def _absorver_diagnostico(rotulo_escolhido: str, resultado: Dict[str, Any]) -> None:
    if not st.session_state.get("admin_autenticado"):
        st.error("Somente o administrador pode confirmar casos e treinar a IA.")
        return
    try:
        with st.spinner("Salvando o caso e retreinando a IA..."):
            absorcao = absorver_caso(
                rotulo_escolhido,
                np.array(resultado["features_audio"], dtype=np.float64),
                resultado["texto_laudo"],
                resultado["wav_bytes"],
                hash_video=resultado.get("hash_video", ""),
                nome_arquivo=resultado.get("nome_arquivo", ""),
                modelo_maquina=resultado.get("modelo_maquina", ""),
            )
    except PipelineError as exc:
        st.error(f"Não foi possível absorver o caso: {exc}")
        return
    except Exception as exc:
        logger.exception("Erro inesperado ao absorver caso de diagnóstico")
        st.error(f"Ocorreu um erro inesperado ao treinar a nova amostra: {exc}")
        return

    if absorcao.duplicado_de:
        _flash("diag", "info",
               f"Este vídeo já estava na base como **{_md(absorcao.rotulo)}** ({absorcao.duplicado_de}); "
               "nada foi alterado.")
    elif absorcao.treinou:
        _flash("diag", "success",
               f"Caso registrado como **{_md(absorcao.rotulo)}** ({absorcao.registro_id}). "
               "A IA absorveu este caso e ficou mais inteligente!",
               toast="Caso registrado e IA retreinada!")
    else:
        _flash("diag", "success",
               f"Caso registrado como **{_md(absorcao.rotulo)}** ({absorcao.registro_id}).",
               toast="Caso registrado!")
        _flash("diag", "info", f"A IA ainda não foi (re)treinada: {_md(absorcao.motivo_sem_treino)}")
    if absorcao.classes_aguardando:
        _flash("diag", "caption",
               f"Classes em acumulação (< {MIN_SAMPLES_PER_CLASS} amostras): "
               + ", ".join(f"{_md(r)} ({n})" for r, n in absorcao.classes_aguardando.items()))

    _descartar_diagnostico()
    st.rerun()


def _formatar_pecas(pecas: List[Tuple[str, int]]) -> str:
    return ", ".join(f"{_md(p)} ({n} caso{'s' if n > 1 else ''})" for p, n in pecas[:3])


def _render_proposta_peca(resultado: Dict[str, Any], probs_relevantes: List[Tuple[str, float]]) -> None:
    """Para os problemas mais prováveis, mostra as peças que a Alimentação
    registrou para o MESMO modelo de máquina (ou, se não houver, para outros)."""
    modelo = resultado.get("modelo_maquina") or "não informado"
    with st.container(border=True):
        st.markdown(f"**Proposta de peça — modelo {_md(modelo)}**")
        if probs_relevantes and eh_referencia(probs_relevantes[0][0]):
            st.caption(
                f"O som mais parecido é o da máquina funcionando normal ({ROTULO_REFERENCIA}): "
                "nenhuma troca de peça é indicada."
            )
            return
        problemas = [(r, p) for r, p in probs_relevantes if not eh_referencia(r)][:3]
        if not problemas:
            st.caption("Nenhum problema provável para propor peça.")
            return
        for rotulo, prob in problemas:
            pecas = (resultado.get("pecas") or {}).get(rotulo) or {}
            mesmo, outros = pecas.get("mesmo_modelo") or [], pecas.get("outros_modelos") or []
            titulo = f"**{_md(rotulo)}** ({prob:.0%})"
            if mesmo:
                st.caption(f"{titulo} — neste modelo: {_formatar_pecas(mesmo)}")
            elif outros:
                st.caption(f"{titulo} — nenhuma peça registrada para este modelo; em outros modelos: {_formatar_pecas(outros)}")
            else:
                st.caption(f"{titulo} — nenhuma peça registrada ainda para este problema.")


def _render_textos_defeito(probs_relevantes: List[Tuple[str, float]]) -> None:
    """Causas, o que verificar e o que fazer da categoria mais provável,
    escritos pelo administrador em Alimentação → Textos dos defeitos."""
    if not probs_relevantes:
        return
    principal = probs_relevantes[0][0]
    textos = textos_da_categoria(principal)
    with st.container(border=True):
        st.markdown(f"**O que fazer — {_md(nome_exibicao(principal))}**")
        if not textos:
            st.caption("Ainda não há textos para esta categoria. O administrador pode escrevê-los em "
                       "Alimentação → Textos dos defeitos.")
            return
        for campo, titulo in CAMPOS_TEXTO.items():
            conteudo = str(textos.get(campo, "")).strip()
            if conteudo:
                st.markdown(f"**{titulo}**  \n" + _md(conteudo).replace("\n", "  \n"))


def render_tab_diagnostico(estado_modelo: Optional[EstadoModelo], status: StatusBase) -> None:
    st.subheader("Diagnóstico Inteligente (Operador)")
    _mostrar_flash("diag")
    geracao = st.session_state["diag_geracao"]
    resultado: Optional[Dict[str, Any]] = st.session_state.get("diag_resultado")

    if resultado is None:
        if estado_modelo is None:
            st.info(
                "O diagnóstico fica disponível depois do primeiro treino da IA. "
                + (f"Para treinar: {_md(status.motivo_sem_treino)}" if status.motivo_sem_treino else
                   "Use o botão “Treinar IA agora” na aba de Alimentação.")
            )
            return

        modelos_existentes = sorted(m for m in status.contagens_modelos if m != "Não informado")
        with st.form(f"form_diagnostico_{geracao}", clear_on_submit=False):
            arquivo_video = st.file_uploader(
                "Vídeo desconhecido", type=TIPOS_ACEITOS, key=f"upload_diagnostico_{geracao}"
            )
            modelo_maquina = st.text_input(
                "Modelo da máquina", key=f"modelo_diagnostico_{geracao}",
                placeholder="Digite o modelo da máquina gravada",
                help=("Modelos já cadastrados: " + ", ".join(modelos_existentes)) if modelos_existentes else None,
            )
            texto_laudo = st.text_area(
                "Texto do laudo do respectivo vídeo", key=f"texto_diagnostico_{geracao}", height=120,
                placeholder="Descreva o som e o que foi observado. Nomes de problemas são ignorados pela IA.",
            )
            enviado = st.form_submit_button("Gerar Diagnóstico")

        if not enviado:
            return
        if arquivo_video is None:
            st.warning("Envie um vídeo desconhecido antes de gerar o diagnóstico.")
            return
        if not limpar_texto(modelo_maquina):
            st.warning("Informe o modelo da máquina.")
            return
        if not limpar_texto(texto_laudo):
            st.warning("O texto do laudo não pode estar vazio.")
            return

        barra = st.progress(0.0, text="Preparando...")
        try:
            novo = gerar_diagnostico(
                arquivo_video.getvalue(), arquivo_video.name, texto_laudo, estado_modelo,
                etapa=lambda texto, fracao: barra.progress(min(fracao, 1.0), text=_md(f"{texto}...")),
                modelo_maquina=modelo_maquina,
            )
        except PipelineError as exc:
            barra.empty()
            st.error(f"Não foi possível gerar o diagnóstico: {exc}")
            return
        except Exception as exc:
            logger.exception("Erro inesperado ao gerar diagnóstico")
            barra.empty()
            st.error(f"Ocorreu um erro inesperado ao gerar o diagnóstico: {exc}")
            return
        st.session_state["diag_resultado"] = novo
        st.rerun()
        return

    # ---- Passo B: resultado dinâmico (persistido em session_state) ----
    probs: Dict[str, float] = resultado["probs"]
    metadados = resultado.get("metadados") or {}
    descricao = f"Arquivo analisado: {_md(resultado.get('nome_arquivo', ''))}"
    if resultado.get("modelo_maquina"):
        descricao += f" · modelo {_md(resultado['modelo_maquina'])}"
    if metadados:
        descricao += f" · {metadados['duracao_s']:.1f} s de áudio @ {metadados['sample_rate']} Hz"
    if resultado.get("ratio_fala") is not None and resultado["ratio_fala"] >= MIN_SPEECH_RATIO:
        descricao += (" · voz humana removida" if resultado.get("demucs_aplicado")
                      else " · voz humana detectada (não removida)")
    st.caption(descricao + ".")
    for aviso in resultado.get("avisos") or []:
        st.caption(f"↳ Atenção: {_md(aviso)}")
    if resultado.get("termos_retirados"):
        st.caption(
            "↳ Trechos do laudo ignorados pela IA (nomes de categorias cadastradas): "
            + ", ".join(_md(t) for t in resultado["termos_retirados"]) + "."
        )

    admin = bool(st.session_state.get("admin_autenticado"))
    probs_relevantes = [
        (rotulo, p) for rotulo, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True) if p > 0.01
    ]
    rotulo_escolhido: Optional[str] = None

    if admin:
        st.markdown("**Clique no problema mais provável ou envie outro para treinar a IA.**")
        if not probs_relevantes:
            st.warning("Nenhuma classe atingiu probabilidade acima de 1%. Use o campo 'Outro problema' abaixo.")
        for i, (rotulo, prob) in enumerate(probs_relevantes):
            if st.button(f"{nome_exibicao(rotulo)} ({prob:.0%})", key=f"botao_prob_{geracao}_{i}", width="stretch"):
                rotulo_escolhido = rotulo
    else:
        st.markdown("**Problemas mais prováveis**")
        if not probs_relevantes:
            st.warning("Nenhuma classe atingiu probabilidade acima de 1%.")
        for rotulo, prob in probs_relevantes:
            st.progress(min(max(prob, 0.0), 1.0), text=f"{_md(nome_exibicao(rotulo))} — {prob:.0%}")
    st.caption(
        "A porcentagem é a confiança interna do modelo, não uma certeza — confirme com a inspeção do equipamento."
    )

    _render_proposta_peca(resultado, probs_relevantes)
    _render_textos_defeito(probs_relevantes)

    if not admin:
        st.caption("A confirmação do diagnóstico e o treino da IA são feitos pelo administrador.")
        st.button("Analisar outro vídeo", key=f"diag_descartar_{geracao}", on_click=_descartar_diagnostico)
        return

    st.markdown("---")
    col_txt, col_btn = st.columns([3, 1], vertical_alignment="bottom")
    with col_txt:
        outro_rotulo = st.text_input(
            "Outro problema", key=f"input_outro_rotulo_{geracao}", placeholder="Digite o problema aqui"
        )
    with col_btn:
        enviar_outro = st.button("Treinar Nova Classe", key=f"botao_outro_rotulo_{geracao}", width="stretch")

    if enviar_outro:
        if not limpar_texto(outro_rotulo):
            st.warning("Digite o nome do problema antes de treinar a nova classe.")
        else:
            rotulo_escolhido = limpar_texto(outro_rotulo)

    st.button("Descartar e analisar outro vídeo", key=f"diag_descartar_{geracao}", on_click=_descartar_diagnostico)

    if rotulo_escolhido:
        _absorver_diagnostico(rotulo_escolhido, resultado)
