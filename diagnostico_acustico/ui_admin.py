"""Interface: aba Alimentação (Admin)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .config import (
    MAX_VIDEO_DURATION_WARNING_S,
    MIN_SAMPLES_PER_CLASS,
    MIN_SPEECH_RATIO,
    ROTULO_REFERENCIA,
    TIPOS_ACEITOS,
    logger,
)
from .dataset import eh_referencia, nome_exibicao, normalizar_modelo, normalizar_rotulo
from .erros import PipelineError
from .pipeline import (
    EntradaCaso,
    ItemLote,
    StatusBase,
    alimentar_sistema_em_lote,
    excluir_casos,
    retreinar_agora,
    validar_entradas_admin,
)
from .texto import limpar_texto
from .treino import EstadoModelo
from .ui_comum import _descricao_modelo, _flash, _formatar_data, _md, _mostrar_flash, _novo_id_caso


# ---------------------------------------------------------------------- Admin

def _adicionar_caso_admin() -> None:
    st.session_state["admin_casos"].append(_novo_id_caso())


def _limpar_widgets_caso(caso_id: str) -> None:
    for prefixo in (
        "admin_video_", "admin_texto_", "admin_rotulo_", "admin_remover_",
        "admin_referencia_", "admin_modelo_", "admin_peca_",
    ):
        st.session_state.pop(f"{prefixo}{caso_id}", None)
    st.session_state["admin_erros_casos"].pop(caso_id, None)


def _remover_caso_admin(caso_id: str) -> None:
    casos = st.session_state["admin_casos"]
    if caso_id in casos and len(casos) > 1:
        casos.remove(caso_id)
        _limpar_widgets_caso(caso_id)


def _fechar_resultado_admin() -> None:
    st.session_state["admin_resultado"] = None


def _atualizar_casos_apos_envio(itens: List[Dict[str, Any]]) -> None:
    """Some com os blocos cujos vídeos foram todos cadastrados (ou já estavam
    na base) e mantém na tela apenas os que tiveram erro, com o motivo, para
    o usuário corrigir e reenviar. Sempre sobra ao menos um bloco vazio."""
    por_caso: Dict[str, List[Dict[str, Any]]] = {}
    for item in itens:
        por_caso.setdefault(item["caso_id"], []).append(item)

    manter: List[str] = []
    erros: Dict[str, List[str]] = {}
    for caso_id in st.session_state["admin_casos"]:
        itens_caso = por_caso.get(caso_id)
        if not itens_caso:  # bloco que não participou do envio (vazio)
            manter.append(caso_id)
            continue
        falhas = [i for i in itens_caso if i["status"] == "erro"]
        completos = len(itens_caso) >= itens_caso[0]["total_no_caso"]
        if falhas or not completos:
            manter.append(caso_id)
            erros[caso_id] = [f"{i['arquivo']}: {i['erro']}" for i in falhas]
        else:
            _limpar_widgets_caso(caso_id)

    st.session_state["admin_casos"] = manter or [_novo_id_caso()]
    st.session_state["admin_erros_casos"] = {k: v for k, v in erros.items() if v}


def _recuperar_envio_interrompido() -> None:
    """Se um envio foi interrompido no meio (ex.: o usuário clicou em outro
    botão durante o processamento), os vídeos concluídos até ali já estão
    salvos; aqui eles viram o "resultado do último envio"."""
    parcial = st.session_state.get("admin_parcial")
    st.session_state["admin_parcial"] = None
    if not parcial:
        return
    st.session_state["admin_resultado"] = {
        "itens": parcial, "treinou": False, "classes_treinadas": [], "acuracia_cv": None,
        "n_folds_cv": 0, "n_amostras_treino": 0, "classes_aguardando": {}, "erro_treino": None,
        "motivo_sem_treino": None, "interrompido": True,
    }
    st.session_state["admin_toast_pendente"] = True
    _atualizar_casos_apos_envio(parcial)


def _executar_alimentacao(entradas: List[EntradaCaso]) -> None:
    st.session_state["admin_resultado"] = None
    st.session_state["admin_parcial"] = []
    barra = st.progress(0.0, text="Preparando o processamento...")

    def progresso(fracao: float, texto: str) -> None:
        barra.progress(float(min(max(fracao, 0.0), 1.0)), text=_md(texto))

    def ao_processar_item(item: ItemLote) -> None:
        st.session_state["admin_parcial"].append(asdict(item))

    try:
        resultado = alimentar_sistema_em_lote(entradas, progresso=progresso, ao_processar_item=ao_processar_item)
    except PipelineError as exc:
        barra.empty()
        _recuperar_envio_interrompido()
        st.error(f"Não foi possível alimentar o sistema: {exc}")
        return
    except Exception as exc:
        logger.exception("Erro inesperado ao alimentar o sistema em lote")
        barra.empty()
        _recuperar_envio_interrompido()
        st.error(f"Ocorreu um erro inesperado ao alimentar o sistema: {exc}")
        return

    st.session_state["admin_parcial"] = None
    st.session_state["admin_resultado"] = resultado.como_dict()
    st.session_state["admin_toast_pendente"] = True
    _atualizar_casos_apos_envio(st.session_state["admin_resultado"]["itens"])
    st.rerun()


def _linha_item(item: Dict[str, Any]) -> str:
    # Sem "Caso N": depois do envio os blocos são renumerados, e o número confundiria.
    partes = [f"{_md(item['arquivo'])} → **{_md(item['rotulo'])}**"]
    if item.get("modelo"):
        partes.append(f"modelo {_md(item['modelo'])}")
    if item.get("duracao_s") is not None:
        partes.append(f"{item['duracao_s']:.1f} s de áudio")
    ratio = item.get("ratio_fala")
    if ratio is not None:
        if item.get("demucs_aplicado"):
            partes.append(f"voz humana em {ratio:.0%} do áudio, removida")
        elif ratio >= MIN_SPEECH_RATIO:
            partes.append(f"voz humana em {ratio:.0%} do áudio (não removida)")
        else:
            partes.append("sem voz humana")
    if item.get("registro_id"):
        partes.append(str(item["registro_id"]))
    return " · ".join(partes)


def render_resultado_admin() -> None:
    """Confirmação persistente do último envio: fica visível até o próximo
    envio (ou até o usuário fechar), e também aparece como notificação."""
    resultado = st.session_state.get("admin_resultado")
    if not resultado:
        return

    itens = resultado["itens"]
    ok = [i for i in itens if i["status"] == "ok"]
    duplicados = [i for i in itens if i["status"] == "duplicado"]
    falhas = [i for i in itens if i["status"] == "erro"]

    if st.session_state.get("admin_toast_pendente"):
        st.session_state["admin_toast_pendente"] = False
        if ok:
            st.toast(f"{len(ok)} vídeo(s) recebido(s) e salvo(s) na base.")
        elif falhas:
            st.toast("Nenhum vídeo foi cadastrado — veja os detalhes abaixo.")
        elif duplicados:
            st.toast("Os vídeos enviados já estavam na base.")

    with st.container(border=True):
        st.markdown("**Resultado do último envio**")
        if resultado.get("interrompido"):
            st.warning(
                "O envio foi interrompido antes de terminar. Os vídeos listados abaixo como "
                "cadastrados já estão salvos; reenvie os demais (duplicatas são ignoradas)."
            )

        if ok:
            st.success(f"{len(ok)} vídeo(s) recebido(s), processado(s) e salvo(s) na base de conhecimento.")
            for item in ok:
                st.caption(_linha_item(item))
                for aviso in item.get("avisos") or []:
                    st.caption(f"↳ Atenção: {_md(aviso)}")

        if duplicados:
            st.info(f"{len(duplicados)} vídeo(s) ignorado(s) porque já estavam na base:")
            for item in duplicados:
                st.caption(f"{_md(item['arquivo'])} — já cadastrado como **{_md(item['rotulo'])}** ({item['duplicado_de']})")

        if falhas:
            st.error(f"{len(falhas)} vídeo(s) não puderam ser processados (os casos continuam na tela para correção):")
            for item in falhas:
                st.caption(f"{_md(item['arquivo'])}: {_md(item['erro'])}")

        if ok and not resultado.get("interrompido"):
            if resultado.get("erro_treino"):
                st.error(f"Os vídeos foram salvos, mas o treino da IA falhou: {_md(resultado['erro_treino'])}")
            elif resultado["treinou"]:
                texto = f"IA treinada com sucesso com {resultado['n_amostras_treino']} casos!"
                if resultado.get("acuracia_cv") is not None:
                    texto += (
                        f" Acurácia estimada: {resultado['acuracia_cv']:.0%} "
                        f"(validação cruzada em {resultado['n_folds_cv']} partes)."
                    )
                st.success(texto)
                st.caption(f"Classes treinadas: {_md(', '.join(resultado['classes_treinadas']))}")
            else:
                st.info(f"Os dados foram salvos, mas a IA ainda não foi treinada: {_md(resultado['motivo_sem_treino'])}")

        if resultado.get("classes_aguardando"):
            st.caption(
                f"Classes em acumulação (< {MIN_SAMPLES_PER_CLASS} amostras, fora do treino por enquanto): "
                + ", ".join(f"{_md(r)} ({n})" for r, n in resultado["classes_aguardando"].items())
            )
        st.button("Fechar", key="admin_fechar_resultado", on_click=_fechar_resultado_admin)


def _executar_treino_manual() -> None:
    try:
        with st.spinner("Treinando a IA (LightGBM)..."):
            treinou, resultado_treino, motivo = retreinar_agora()
    except PipelineError as exc:
        st.error(f"Não foi possível treinar a IA: {exc}")
        return
    if treinou and resultado_treino is not None:
        texto = f"IA treinada com sucesso com {resultado_treino.n_amostras_treino} casos!"
        if resultado_treino.acuracia_cv is not None:
            texto += f" Acurácia estimada: {resultado_treino.acuracia_cv:.0%}."
        _flash("base", "success", texto, toast="IA treinada com sucesso!")
    else:
        _flash("base", "info", f"A IA não foi treinada: {motivo}")
    st.rerun()


def _render_gestao_casos(status: StatusBase) -> None:
    tabela = status.tabela_casos
    if tabela is None or tabela.empty:
        st.caption("Nenhum caso cadastrado ainda.")
        return
    exibicao = pd.DataFrame({
        "ID": tabela["UUID"],
        "Problema": tabela["rotulo"],
        "Modelo da máquina": tabela["modelo_maquina"].replace({"": "—"}),
        "Peça": tabela["peca"].replace({"": "—"}),
        "Origem": tabela["origem"].replace({"admin": "Alimentação", "diagnostico": "Diagnóstico", "": "—"}),
        "Arquivo enviado": tabela["arquivo_original"].replace({"": "—"}),
        "Cadastrado em": tabela["criado_em"].map(_formatar_data),
    }).iloc[::-1].reset_index(drop=True)  # mais recentes primeiro

    evento = st.dataframe(
        exibicao, hide_index=True, on_select="rerun", selection_mode="multi-row",
        key=f"admin_tabela_casos_{st.session_state['admin_tabela_geracao']}",
    )
    linhas = list(evento.selection.rows) if evento is not None else []
    if not linhas:
        st.caption("Selecione linhas na tabela para excluir casos cadastrados por engano.")
        return

    selecionados = exibicao.iloc[linhas]["ID"].tolist()
    st.caption(
        f"{len(selecionados)} caso(s) selecionado(s). A exclusão apaga o áudio, o laudo e a linha do "
        "dataset; em seguida a IA é retreinada com os casos restantes."
    )
    confirmar = st.checkbox(
        "Confirmo que quero excluir definitivamente os casos selecionados",
        key=f"admin_confirmar_exclusao_{st.session_state['admin_tabela_geracao']}",
    )
    if st.button("Excluir selecionados", disabled=not confirmar, key="admin_excluir_casos"):
        try:
            with st.spinner("Excluindo casos e retreinando a IA..."):
                resumo = excluir_casos(selecionados)
        except PipelineError as exc:
            st.error(f"Não foi possível excluir: {exc}")
            return
        texto = f"{resumo['n_removidos']} caso(s) excluído(s)."
        if resumo["treinou"]:
            texto += " A IA foi retreinada com os casos restantes."
        else:
            texto += f" A IA ficou sem dados suficientes e voltou ao estado inicial: {resumo['motivo_sem_treino']}"
        _flash("base", "success", texto, toast=f"{resumo['n_removidos']} caso(s) excluído(s).")
        st.session_state["admin_tabela_geracao"] += 1
        st.rerun()


def render_painel_base(status: StatusBase, estado_modelo: Optional[EstadoModelo]) -> None:
    """Mostra o que já foi aprendido e quanto falta para cada problema entrar
    no treino -- deixa claro POR QUE a IA ainda não treinou."""
    with st.container(border=True):
        st.markdown("**Base de conhecimento**")
        _mostrar_flash("base")
        if status.erro:
            st.error(_md(status.erro))
            return

        st.caption(
            f"{status.total_casos} caso(s) cadastrado(s) · {len(status.contagens)} problema(s) diferente(s)"
        )
        if estado_modelo is not None:
            st.caption(f"IA {_descricao_modelo(estado_modelo.metadata)}.")
        else:
            st.caption("IA ainda não treinada.")

        for rotulo, n in sorted(status.contagens.items(), key=lambda kv: (-kv[1], kv[0])):
            if n >= MIN_SAMPLES_PER_CLASS:
                texto = f"{_md(nome_exibicao(rotulo))} — {n} casos · pronto para o treino"
            else:
                texto = f"{_md(nome_exibicao(rotulo))} — {n}/{MIN_SAMPLES_PER_CLASS} casos · faltam {MIN_SAMPLES_PER_CLASS - n}"
            st.progress(min(n / MIN_SAMPLES_PER_CLASS, 1.0), text=texto)

        if status.contagens_modelos:
            st.caption(
                "Casos por modelo de máquina: "
                + " · ".join(f"{_md(m)} ({n})" for m, n in sorted(status.contagens_modelos.items(), key=lambda kv: (-kv[1], kv[0])))
            )

        if not status.pode_treinar and status.motivo_sem_treino:
            st.caption(f"Para treinar: {_md(status.motivo_sem_treino)}")
        if status.modelo_desatualizado:
            st.info("Há casos cadastrados que ainda não foram incorporados à IA.")
            if st.button("Treinar IA agora", key="admin_treinar_agora", type="primary"):
                _executar_treino_manual()

        with st.expander(f"Gerenciar casos cadastrados ({status.total_casos})"):
            _render_gestao_casos(status)


def render_tab_admin(status: StatusBase, estado_modelo: Optional[EstadoModelo]) -> None:
    _recuperar_envio_interrompido()

    st.subheader("Alimentação do Sistema (Admin)")
    st.caption(
        "Cadastre um ou mais vídeos conhecidos, cada um com o laudo textual e o nome exato "
        "do problema. Use o botão “+ Novo caso” para alimentar vários problemas de uma só vez; "
        "num mesmo caso você pode enviar vários vídeos do mesmo problema."
    )
    st.info(
        f"Não há limite de tamanho de arquivo, mas se o vídeo passar de "
        f"{MAX_VIDEO_DURATION_WARNING_S // 60} minutos, corte para os melhores momentos e "
        f"mantenha o vídeo dentro desse limite antes de enviar -- vídeos mais curtos e "
        f"focados no defeito geram um treino mais rápido e mais preciso."
    )

    render_painel_base(status, estado_modelo)

    rotulos_existentes = sorted(status.contagens)
    if rotulos_existentes:
        st.caption(
            "Problemas já cadastrados (use o mesmo nome para somar casos ao mesmo problema; "
            "maiúsculas, minúsculas e acentos são unificados automaticamente): "
            + " · ".join(_md(r) for r in rotulos_existentes)
            + ". Quando esses nomes aparecem num laudo, a IA os ignora."
        )
    modelos_existentes = sorted(m for m in status.contagens_modelos if m != "Não informado")
    if modelos_existentes:
        st.caption("Modelos de máquina já cadastrados: " + " · ".join(_md(m) for m in modelos_existentes))

    casos: List[str] = st.session_state["admin_casos"]
    entradas: List[EntradaCaso] = []
    for numero, caso_id in enumerate(casos, start=1):
        with st.container(border=True):
            col_titulo, col_remover = st.columns([5, 1], vertical_alignment="center")
            with col_titulo:
                st.markdown(f"**Caso {numero}**")
            with col_remover:
                if len(casos) > 1:
                    st.button("Remover", key=f"admin_remover_{caso_id}",
                              on_click=_remover_caso_admin, args=(caso_id,))

            erros_caso = st.session_state["admin_erros_casos"].get(caso_id)
            if erros_caso:
                st.error("Este caso não foi cadastrado. Corrija e clique em Alimentar Sistema de novo:\n\n"
                         + "\n\n".join(f"• {_md(e)}" for e in erros_caso))

            arquivos = st.file_uploader(
                "Vídeo(s) conhecido(s) do mesmo problema", type=TIPOS_ACEITOS,
                accept_multiple_files=True, key=f"admin_video_{caso_id}",
                help="Formatos aceitos: " + ", ".join(TIPOS_ACEITOS)
                     + ". Vários vídeos no mesmo caso compartilham o laudo e o rótulo abaixo.",
            )
            texto_laudo = st.text_area(
                "Texto do laudo do respectivo vídeo", key=f"admin_texto_{caso_id}", height=100,
                placeholder="Descreva o som e o que foi observado. Nomes de problemas são ignorados pela IA.",
            )
            modelo = st.text_input(
                "Modelo da máquina", key=f"admin_modelo_{caso_id}",
                placeholder="Digite o modelo da máquina gravada",
            )
            modelo_limpo = limpar_texto(modelo)
            if modelo_limpo:
                canonico_modelo = normalizar_modelo(modelo_limpo, modelos_existentes)
                if canonico_modelo != modelo_limpo:
                    st.caption(f"Será somado ao modelo já cadastrado **{_md(canonico_modelo)}**.")
                elif canonico_modelo not in modelos_existentes:
                    st.caption("Modelo novo — será criado na base.")
            referencia = st.checkbox(
                "Vídeo(s) de referência: máquina funcionando normal, sem defeito",
                key=f"admin_referencia_{caso_id}",
            )
            peca = ""
            if referencia:
                rotulo = ROTULO_REFERENCIA
                st.caption(f"Será cadastrado na categoria **{ROTULO_REFERENCIA}** (funcionamento normal).")
            else:
                rotulo = st.text_input(
                    "Nome exato do problema (rótulo)", key=f"admin_rotulo_{caso_id}"
                )
                rotulo_limpo = limpar_texto(rotulo)
                if rotulo_limpo:
                    canonico = normalizar_rotulo(rotulo_limpo, rotulos_existentes)
                    if eh_referencia(canonico):
                        st.caption(f"Será cadastrado na categoria **{ROTULO_REFERENCIA}** (funcionamento normal).")
                    elif canonico != rotulo_limpo:
                        st.caption(f"Será somado ao problema já cadastrado **{_md(canonico)}**.")
                    elif canonico not in rotulos_existentes:
                        st.caption("Problema novo — será criado na base.")
                peca = st.text_input(
                    "Peça ou componente indicado (opcional)", key=f"admin_peca_{caso_id}",
                    placeholder="Ex.: rolamento 6203-2RS do motor",
                    help="Aparece como proposta de peça no diagnóstico de máquinas deste mesmo modelo.",
                )
            entradas.append(EntradaCaso(
                caso_id=caso_id, numero=numero, arquivos=list(arquivos or []),
                texto=texto_laudo or "", rotulo=rotulo or "", modelo=modelo or "", peca=peca or "",
            ))

    st.button("+ Novo caso", key="admin_novo_caso", on_click=_adicionar_caso_admin)

    st.markdown("---")

    if st.button("Alimentar Sistema", type="primary", width="stretch", key="admin_alimentar"):
        # A Referência preenche o rótulo sozinha, então ela não conta como "caso preenchido".
        preenchidas = [
            e for e in entradas
            if e.arquivos or limpar_texto(e.texto) or limpar_texto(e.modelo)
            or (limpar_texto(e.rotulo) and not eh_referencia(e.rotulo))
        ]
        if not preenchidas:
            st.warning("Preencha ao menos um caso (vídeo + texto + modelo + rótulo) antes de alimentar o sistema.")
        elif problemas := validar_entradas_admin(preenchidas):
            st.warning("Complete os campos antes de alimentar o sistema:\n\n"
                       + "\n\n".join(f"• {p}" for p in problemas))
        else:
            _executar_alimentacao(preenchidas)

    render_resultado_admin()
