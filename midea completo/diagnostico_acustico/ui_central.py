"""Interface: partes do modo administrador ligadas ao app do celular.

    Caixa de entrada           casos gravados pelos técnicos, para aprovar ou rejeitar
    Textos dos defeitos        causas, o que verificar e o que fazer de cada categoria
    Celulares e sincronização  troca com a caixa postal, pacote para os celulares,
                               contas dos técnicos e configuração
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from . import caixa_entrada
from .conhecimento import CAMPOS_TEXTO, salvar_textos_categoria, textos_da_categoria
from .config import ROTULO_REFERENCIA, logger
from .dataset import eh_referencia, nome_exibicao
from .erros import PipelineError
from .pacote import gerar_pacote, listar_publicacoes
from .pipeline import StatusBase
from .sincronizacao import (
    ConfigSincronizacao,
    SincronizacaoError,
    CaixaPostal,
    carregar_config,
    carregar_estado,
    codigo_conexao,
    pendencias,
    salvar_config,
    sincronizar,
    testar_conexao,
)
from .texto import limpar_texto
from .treino import EstadoModelo
from .ui_comum import _flash, _formatar_data, _md, _mostrar_flash


def _chave_widget(*partes: str) -> str:
    return "w_" + hashlib.sha1("|".join(partes).encode("utf-8")).hexdigest()[:12]


def _formatar_probs(probs: Optional[Dict[str, float]], limite: int = 3) -> str:
    if not probs:
        return "—"
    ordem = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:limite]
    return " · ".join(f"{_md(nome_exibicao(r))} {p:.0%}" for r, p in ordem)


# ------------------------------------------------------------------ sincronizar

def executar_sincronizacao(estado_modelo: Optional[EstadoModelo], area: str = "celulares") -> bool:
    """Roda uma troca com a caixa postal mostrando o andamento; o resultado
    aparece como mensagem na área indicada depois do st.rerun()."""
    barra = st.progress(0.0, text="Sincronizando com a caixa postal...")
    try:
        resumo = sincronizar(estado_modelo, progresso=lambda f, t: barra.progress(min(max(f, 0.0), 1.0), text=_md(t)))
    except PipelineError as exc:
        barra.empty()
        _flash(area, "warning", f"Não foi possível sincronizar agora: {_md(exc)} Tudo continua funcionando "
                                 "sem internet; as pendências seguem guardadas para a próxima troca.")
        return False
    barra.empty()
    partes = []
    base = resumo.get("base")
    if base:
        texto_base = (f"base compartilhada: {base['enviados']} caso(s) enviado(s) e {base['recebidos']} recebido(s) "
                      "de outros computadores")
        if base["excluidos"]:
            texto_base += f", {base['excluidos']} excluído(s)"
        if base["textos_recebidos"]:
            texto_base += f", {base['textos_recebidos']} texto(s) de defeito atualizado(s)"
        partes.append(texto_base)
    if resumo.get("central", True):
        partes += [
            f"{resumo['recebidos']} caso(s) dos celulares recebido(s)",
            f"{resumo['analises_enviadas']} análise(s) e {resumo['respostas_enviadas']} decisão(ões) enviada(s) aos técnicos",
            f"{resumo['publicacoes_enviadas']} pacote(s) enviado(s) aos celulares",
        ]
    _flash(area, "success", "Sincronizado: " + "; ".join(partes) + ".", toast="Sincronização concluída.")
    for aviso in resumo["avisos"]:
        _flash(area, "caption", f"↳ Atenção: {_md(aviso)}")
    return True


def sincronizar_ao_entrar(estado_modelo: Optional[EstadoModelo]) -> None:
    """Logo depois do login do administrador, tenta uma troca automática (uma
    vez por sessão). Sem internet, não atrapalha: só avisa."""
    if not st.session_state.pop("sincronizar_ao_entrar", False) or not carregar_config().configurada:
        return
    if executar_sincronizacao(estado_modelo):
        st.rerun()


def render_sincronizacao_sidebar(estado_modelo: Optional[EstadoModelo]) -> None:
    if not st.session_state.get("admin_autenticado"):
        return
    n_caixa = len(caixa_entrada.casos_pendentes())
    configurada = carregar_config().configurada
    if not configurada and not n_caixa:
        return
    st.markdown("### Sincronização")
    _mostrar_flash("sidebar_sync")
    if n_caixa:
        st.caption(f"**{n_caixa} caso(s)** dos celulares esperando decisão (Alimentação → Caixa de entrada).")
    if not configurada:
        return
    estado = carregar_estado()
    pend = pendencias()
    st.caption(f"Última troca: {_formatar_data(estado.get('ultima_sincronizacao'))}")
    if not carregar_config().central:
        st.caption("Este computador troca só a base de casos com os outros (não é a Central dos celulares).")
    if any(pend.values()):
        st.caption(f"Esperando internet: {pend['analises']} análise(s), {pend['respostas']} decisão(ões), "
                   f"{pend['publicacoes']} pacote(s).")
    if st.button("Sincronizar agora", key="sidebar_sincronizar", width="stretch"):
        executar_sincronizacao(estado_modelo, area="sidebar_sync")
        st.rerun()


# ------------------------------------------------------------- caixa de entrada

def _analisar(caso_id: str, estado_modelo: Optional[EstadoModelo]) -> bool:
    try:
        with st.spinner("Analisando o caso na Central (extração do som, remoção de voz e IA)..."):
            caixa_entrada.analisar_caso(caso_id, estado_modelo)
        return True
    except PipelineError as exc:
        st.error(f"Não foi possível analisar o caso: {exc}")
    except Exception as exc:
        logger.exception("Erro inesperado ao analisar caso da caixa de entrada")
        st.error(f"Erro inesperado ao analisar o caso: {exc}")
    return False


def _render_caso(caso: Dict[str, Any], estado_modelo: Optional[EstadoModelo], categorias: List[str]) -> None:
    caso_id = str(caso["id"])
    estado = caso["estado"]
    with st.container(border=True):
        st.markdown(
            f"**{_md(caso.get('modelo_maquina') or 'Modelo não informado')}** · "
            f"{_md(caso.get('tecnico_nome') or 'técnico')} · {_formatar_data(caso.get('criado_em'))}"
        )
        st.caption(f"Laudo: {_md(caso.get('laudo') or '—')}")
        if caso.get("tem_audio"):
            st.audio(caixa_entrada.audio_do_caso(caso_id), format="audio/wav")

        celular = caso.get("resultado_celular") or {}
        st.caption(f"Resultado no celular: {_formatar_probs(celular.get('probabilidades'))}")
        confirmacao = caso.get("confirmacao_tecnico") or {}
        if confirmacao.get("categoria") or confirmacao.get("observacao"):
            st.caption(
                "O técnico indicou: " + _md(confirmacao.get("categoria") or "—")
                + (f" — {_md(confirmacao['observacao'])}" if confirmacao.get("observacao") else "")
            )

        analise = estado.get("analise_central")
        if analise:
            if analise.get("probabilidades"):
                st.caption(f"Análise da Central (som + laudo): {_formatar_probs(analise['probabilidades'])}")
                if estado.get("analise_enviada_em"):
                    st.caption(f"↳ Esta análise já foi enviada ao técnico em "
                               f"{_formatar_data(estado['analise_enviada_em'])}; falta a sua decisão.")
                else:
                    st.caption("↳ Esta análise vai para o técnico na próxima sincronização, antes da sua decisão.")
            else:
                st.caption("Análise da Central: a IA ainda não foi treinada; só as características do som foram calculadas.")
            if analise.get("termos_retirados"):
                st.caption("↳ Trechos do laudo ignorados pela IA (nomes de categorias cadastradas): "
                           + ", ".join(_md(t) for t in analise["termos_retirados"]) + ".")
            for aviso in analise.get("avisos") or []:
                st.caption(f"↳ Atenção: {_md(aviso)}")
        elif st.button("Analisar na Central", key=f"caixa_analisar_{caso_id}"):
            if _analisar(caso_id, estado_modelo):
                st.rerun()

        sugestao = confirmacao.get("categoria") or celular.get("categoria_sugerida")
        if analise and analise.get("probabilidades"):
            sugestao = sugestao or max(analise["probabilidades"], key=analise["probabilidades"].get)
        opcoes = list(categorias)
        indice = next((i for i, c in enumerate(opcoes) if sugestao and c.casefold() == str(sugestao).casefold()), 0)
        with st.form(f"form_caixa_{caso_id}"):
            categoria = st.selectbox("Categoria do caso", opcoes, index=indice,
                                     format_func=nome_exibicao) if opcoes else None
            outra = st.text_input("Ou digite uma categoria nova", placeholder="Só se não estiver na lista acima")
            peca = st.text_input("Peça ou componente indicado (opcional)")
            observacao = st.text_input("Observação para o técnico (opcional)")
            col_aprovar, col_rejeitar = st.columns(2)
            with col_aprovar:
                aprovar = st.form_submit_button("Aprovar e ensinar a IA", type="primary", width="stretch")
            with col_rejeitar:
                rejeitar = st.form_submit_button("Rejeitar", width="stretch")

        if rejeitar:
            caixa_entrada.rejeitar_caso(caso_id, observacao)
            _flash("caixa", "info", "Caso rejeitado. A resposta vai para o técnico na próxima sincronização.")
            st.rerun()
        if aprovar:
            escolhida = limpar_texto(outra) or limpar_texto(categoria or "")
            if not escolhida:
                st.warning("Escolha a categoria do caso.")
                return
            if not analise and not _analisar(caso_id, estado_modelo):
                return
            try:
                with st.spinner("Salvando o caso na base e retreinando a IA..."):
                    absorcao = caixa_entrada.aprovar_caso(caso_id, escolhida, peca, observacao)
            except PipelineError as exc:
                st.error(f"Não foi possível aprovar o caso: {exc}")
                return
            if absorcao.duplicado_de:
                texto = f"Este áudio já estava na base como **{_md(absorcao.rotulo)}**; nada foi duplicado."
            elif absorcao.treinou:
                texto = f"Caso aprovado como **{_md(absorcao.rotulo)}** e IA retreinada."
            else:
                texto = (f"Caso aprovado como **{_md(absorcao.rotulo)}**. A IA ainda não foi treinada: "
                         f"{_md(absorcao.motivo_sem_treino)}")
            _flash("caixa", "success", texto + " A resposta vai para o técnico na próxima sincronização.",
                   toast="Caso aprovado.")
            st.rerun()


def render_aba_caixa_entrada(estado_modelo: Optional[EstadoModelo], status: StatusBase) -> None:
    st.subheader("Caixa de entrada dos celulares")
    _mostrar_flash("caixa")
    st.caption(
        "Casos gravados pelos técnicos e recebidos na sincronização. A análise da Central vai para o técnico "
        "logo na sincronização seguinte, sem esperar você. A IA só aprende com um caso depois que você o "
        "aprova; a sua decisão vai para o técnico na sincronização seguinte à decisão. Não há prazo para decidir."
    )
    categorias = sorted(set(status.contagens) | {ROTULO_REFERENCIA}, key=lambda c: (eh_referencia(c), c.casefold()))
    pendentes = caixa_entrada.casos_pendentes()
    if not pendentes:
        st.info("Nenhum caso esperando decisão.")
    for caso in pendentes:
        _render_caso(caso, estado_modelo, categorias)

    decididos = [c for c in caixa_entrada.listar_casos() if c["estado"].get("decisao")]
    if decididos:
        with st.expander(f"Casos já decididos ({len(decididos)})"):
            st.dataframe(pd.DataFrame([{
                "Gravado em": _formatar_data(c.get("criado_em")),
                "Técnico": c.get("tecnico_nome") or "—",
                "Modelo da máquina": c.get("modelo_maquina") or "—",
                "Decisão": c["estado"]["decisao"]["status"].capitalize(),
                "Categoria": c["estado"]["decisao"].get("categoria") or "—",
                "Resposta ao técnico": ("enviada em " + _formatar_data(c["estado"]["resposta_enviada_em"]))
                if c["estado"].get("resposta_enviada_em") else "esperando sincronização",
            } for c in reversed(decididos)]), hide_index=True)


# ----------------------------------------------------------- textos dos defeitos

def render_aba_textos(status: StatusBase) -> None:
    st.subheader("Textos de cada defeito")
    _mostrar_flash("textos")
    st.caption(
        "Para cada categoria, escreva as possíveis causas, o que verificar e o que fazer. Os textos "
        "aparecem no resultado do diagnóstico, aqui no site e no celular. No celular, eles chegam no próximo "
        "pacote enviado (aba Celulares e sincronização) e ficam guardados no aparelho, para abrir na hora, "
        "mesmo sem internet."
    )
    categorias = sorted(status.contagens, key=lambda c: (eh_referencia(c), c.casefold()))
    if not categorias:
        st.info("Cadastre as categorias na aba Alimentar primeiro.")
        return
    for categoria in categorias:
        atuais = textos_da_categoria(categoria) or {}
        titulo = nome_exibicao(categoria) + ("" if atuais else " — sem textos")
        with st.expander(titulo):
            with st.form(_chave_widget("textos", categoria)):
                valores = {
                    campo: st.text_area(rotulo, value=atuais.get(campo, ""), height=110,
                                        key=_chave_widget("texto", categoria, campo))
                    for campo, rotulo in CAMPOS_TEXTO.items()
                }
                if st.form_submit_button("Salvar textos"):
                    try:
                        salvar_textos_categoria(categoria, valores)
                    except PipelineError as exc:
                        st.error(str(exc))
                    else:
                        _flash("textos", "success",
                               f"Textos de **{_md(categoria)}** salvos. Para levá-los aos celulares, gere um "
                               "novo pacote na aba Celulares e sincronização.")
                        st.rerun()


# ------------------------------------------------------ celulares e sincronização

def _render_publicar(estado_modelo: Optional[EstadoModelo], configurada: bool) -> None:
    with st.container(border=True):
        st.markdown("**Enviar para os celulares**")
        st.caption(
            "Gera um pacote com a IA atual, os textos dos defeitos, as peças e os modelos de máquina "
            "cadastrados. Os celulares baixam o pacote na próxima vez que tiverem internet. Como o celular "
            "não lê o laudo, a IA dele usa só o som; a análise completa (som + laudo) continua sendo feita "
            "aqui, na caixa de entrada."
        )
        notas = st.text_input("O que mudou nesta versão (opcional)", key="publicar_notas")
        if st.button("Gerar pacote para os celulares", type="primary", key="publicar_gerar"):
            try:
                with st.spinner("Treinando a IA só de som e montando o pacote..."):
                    publicacao = gerar_pacote(notas)
            except PipelineError as exc:
                st.error(f"Não foi possível gerar o pacote: {exc}")
                return
            acerto = publicacao.get("acerto_estimado_so_som")
            texto = f"Pacote versão {publicacao['versao']} gerado ({publicacao['tamanho'] / 1e6:.1f} MB)."
            if acerto is not None:
                texto += f" Acerto estimado só pelo som: {acerto:.0%}."
            _flash("celulares", "success", texto)
            if configurada:
                executar_sincronizacao(estado_modelo)
            else:
                _flash("celulares", "info", "Ele será enviado quando a sincronização for configurada.")
            st.rerun()

        publicacoes = listar_publicacoes()
        if publicacoes:
            st.dataframe(pd.DataFrame([{
                "Versão": p["versao"],
                "Gerado em": _formatar_data(p["criado_em"]),
                "Categorias": ", ".join(p.get("categorias") or []),
                "Acerto só pelo som": f"{p['acerto_estimado_so_som']:.0%}" if p.get("acerto_estimado_so_som") is not None else "—",
                "Situação": ("enviado em " + _formatar_data(p["enviado_em"])) if p.get("enviado_em") else "esperando internet",
                "Notas": p.get("notas") or "—",
            } for p in reversed(publicacoes)]), hide_index=True)


def _render_tecnicos(config: ConfigSincronizacao) -> None:
    with st.expander("Técnicos (contas do app)"):
        if not config.configurada:
            st.caption("Configure a sincronização primeiro.")
            return
        if config.chave_publica:
            st.markdown("**Código de conexão para os celulares**")
            st.caption("No app: Configurações → Conexão com a Central → cole este código. Ele leva só o "
                       "endereço da caixa postal e a chave pública; pode ser enviado aos técnicos.")
            st.code(codigo_conexao(config), language=None, wrap_lines=True)
        else:
            st.caption("Preencha a chave pública na configuração da caixa postal para gerar o código de conexão.")
        st.caption("Cada técnico entra no app com o e-mail e a senha criados aqui (precisa de internet só no primeiro acesso).")
        if st.button("Mostrar técnicos cadastrados", key="tecnicos_listar"):
            try:
                st.session_state["tecnicos_lista"] = CaixaPostal(config).listar_tecnicos()
            except PipelineError as exc:
                st.warning(f"Não foi possível buscar a lista: {exc}")
        lista = st.session_state.get("tecnicos_lista")
        if lista is not None:
            if lista:
                st.dataframe(pd.DataFrame([{
                    "Nome": (u.get("user_metadata") or {}).get("nome") or "—",
                    "E-mail": u.get("email") or "—",
                    "Último acesso": _formatar_data((u.get("last_sign_in_at") or "")[:19]),
                } for u in lista]), hide_index=True)
            else:
                st.caption("Nenhum técnico cadastrado ainda.")
        with st.form("form_novo_tecnico", clear_on_submit=True):
            st.markdown("**Novo técnico**")
            nome = st.text_input("Nome")
            email = st.text_input("E-mail")
            senha = st.text_input("Senha provisória (mínimo 8 caracteres)", type="password")
            if st.form_submit_button("Criar conta do técnico"):
                if not (limpar_texto(nome) and "@" in email and len(senha) >= 8):
                    st.warning("Preencha nome, um e-mail válido e uma senha com pelo menos 8 caracteres.")
                else:
                    try:
                        CaixaPostal(config).criar_tecnico(email, senha, nome)
                    except PipelineError as exc:
                        st.error(f"Não foi possível criar a conta: {exc}")
                    else:
                        st.session_state.pop("tecnicos_lista", None)
                        _flash("celulares", "success", f"Conta de **{_md(nome)}** criada. Passe o e-mail e a "
                                                        "senha provisória ao técnico.")
                        st.rerun()


def _render_configuracao(config: ConfigSincronizacao) -> None:
    with st.expander("Configuração da caixa postal", expanded=not config.configurada):
        st.caption(
            "A caixa postal é um projeto no Supabase. O passo a passo para criá-lo está no arquivo "
            "nuvem/COMO_CONFIGURAR.txt, na pasta do programa. A chave de serviço fica só neste "
            "computador e não vai no backup."
        )
        with st.form("form_config_sync"):
            url = st.text_input("Endereço do projeto (Project URL)", value=config.url,
                                placeholder="https://xxxxxxxx.supabase.co")
            publica = st.text_input("Chave pública (anon / publishable) — é a que vai para o app",
                                    value=config.chave_publica)
            servico = st.text_input(
                "Chave de serviço (service_role / secret)", type="password",
                placeholder="já salva — deixe em branco para manter" if config.chave_servico else "",
            )
            central = st.checkbox(
                "Este computador é a Central (recebe os casos dos celulares e envia a IA para eles)",
                value=config.central if config.configurada else False,
                help="Marque em UM computador só. Os outros computadores do site deixam desmarcado: eles "
                     "trocam só a base de casos (o que um alimenta, todos recebem).",
            )
            if st.form_submit_button("Salvar e testar a conexão"):
                nova = ConfigSincronizacao(url=url, chave_publica=publica.strip(),
                                           chave_servico=servico.strip() or config.chave_servico, central=central)
                if not (nova.url.startswith("https://") and nova.chave_servico):
                    st.warning("Preencha o endereço (começa com https://) e a chave de serviço.")
                    return
                salvar_config(nova)
                try:
                    testar_conexao(nova)
                except SincronizacaoError as exc:
                    _flash("celulares", "warning", f"Configuração salva, mas o teste falhou: {_md(exc)}")
                else:
                    _flash("celulares", "success", "Configuração salva e conexão testada com sucesso.")
                st.rerun()


def render_aba_celulares(estado_modelo: Optional[EstadoModelo]) -> None:
    st.subheader("Celulares e sincronização")
    _mostrar_flash("celulares")
    config = carregar_config()

    with st.container(border=True):
        st.markdown("**Caixa postal na internet**")
        if not config.configurada:
            st.caption("Ainda não configurada (veja Configuração da caixa postal, abaixo). O site continua "
                       "funcionando normalmente sem ela.")
        else:
            estado = carregar_estado()
            pend = pendencias()
            st.caption(f"Última troca: {_formatar_data(estado.get('ultima_sincronizacao'))}")
            if estado.get("ultimo_erro"):
                st.caption(f"↳ Última tentativa ({_formatar_data(estado.get('ultima_tentativa'))}) "
                           f"não concluída: {_md(estado['ultimo_erro'])}")
            st.caption(f"Esperando a próxima troca: {pend['analises']} análise(s) e {pend['respostas']} "
                       f"decisão(ões) para os técnicos, {pend['publicacoes']} pacote(s) para os celulares.")
            if st.button("Sincronizar agora", key="celulares_sincronizar"):
                executar_sincronizacao(estado_modelo)
                st.rerun()
        st.caption("Cada sincronização também troca a base de casos com os outros computadores do site: os casos "
                   "alimentados em qualquer um deles chegam a todos, e cada um treina a IA com a base completa.")

    if config.configurada and not config.central:
        st.info("Este computador não é a Central: os casos dos celulares e os pacotes para eles ficam com o "
                "computador marcado como Central (Configuração da caixa postal, abaixo). Aqui, a sincronização "
                "só troca a base de casos com os outros computadores.")
    else:
        _render_publicar(estado_modelo, config.configurada)
    _render_tecnicos(config)
    _render_configuracao(config)
