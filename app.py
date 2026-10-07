"""
================================================================================
 Sistema de Diagnóstico Acústico de Falhas Mecânicas — app.py
================================================================================
Aplicação Streamlit multimodal (áudio + texto) para diagnosticar falhas
mecânicas em equipamentos a partir de vídeos e laudos textuais.

COMO RODAR (Windows):
    Clique duas vezes em iniciar.bat -- ele instala o que faltar e abre o app.

COMO RODAR (manual / Linux / macOS):
    1. Instale o ffmpeg no sistema operacional (obrigatório, fora do pip):
         - Ubuntu/Debian: sudo apt install ffmpeg
         - macOS (brew):  brew install ffmpeg
         - Windows:       winget install ffmpeg -e --id Gyan.FFmpeg
    2. Crie um ambiente virtual e instale as dependências (ver requirements.txt):
         pip install -r requirements.txt
    3. Baixe o modelo de linguagem em português do spaCy:
         python -m spacy download pt_core_news_sm
    4. Mantenha a pasta ".streamlit/config.toml" ao lado deste app.py -- ela
       remove o limite padrão de 200 MB por upload e define o tema visual.
    5. Rode a aplicação:
         streamlit run app.py

    O iniciar.bat já baixa o modelo de remoção de voz (Demucs, ~80 MB) na
    instalação; rodando à mão, ele é baixado no primeiro vídeo com fala.

Código do programa:
    app.py                  -> esta entrada (é o que o iniciar.bat abre)
    diagnostico_acustico/   -> o sistema, dividido por assunto (áudio, características,
                               laudo, base, treino, caixa de entrada, sincronização, telas...)
    estilo/                 -> cores, fontes e logos (tokens.json), usados pelo site e pelo app
    nuvem/                  -> como preparar a caixa postal na internet (Supabase)

Estrutura criada automaticamente ao lado deste arquivo:
    audios_conhecidos/      -> áudios rotulados,  nomeados [ROTULO]_[UUID].wav
    textos_conhecidos/      -> laudos rotulados,  nomeados [ROTULO]_[UUID].txt
    modelos/                -> modelo_lgbm.joblib, label_encoder.joblib, metadata
    dataset_master.csv      -> tabela mestre (UUID, rótulo, caminhos, metadados, features)
    base_conhecimento.json  -> textos de cada defeito (causas, o que verificar, o que fazer)
    caixa_de_entrada/       -> casos enviados pelos celulares, esperando decisão
    publicacoes/            -> pacotes (IA + textos + peças) gerados para os celulares

O SITE E O APP DO CELULAR:
    Os dois funcionam sem internet. Quando há conexão, o site troca dados com
    a caixa postal (aba Alimentação -> Celulares e sincronização): recebe os
    casos gravados no campo, devolve a decisão e a análise da Central aos
    técnicos e envia aos celulares o pacote com a IA, os textos e as peças.

SENHA DO MODO ADMINISTRADOR:
    A aba "Alimentação (Admin)", a confirmação de casos na aba Diagnóstico e o
    backup só funcionam depois de entrar com a senha de administrador. Na
    primeira vez, o próprio app pede para criar a senha. Ela fica guardada
    cifrada (nunca em texto puro) em .streamlit/senha_admin.json. Se a senha
    for esquecida, apague esse arquivo: o app pedirá uma senha nova.

MIGRAR PARA OUTRO COMPUTADOR (sem perder o treino):
    Use o botão "Baixar backup (.zip)" na barra lateral (modo administrador) e
    extraia o .zip na pasta do app.py na outra máquina. Ele leva
        dataset_master.csv, base_conhecimento.json, audios_conhecidos/,
        textos_conhecidos/, modelos/ e caixa_de_entrada/
    Todos os caminhos salvos no CSV são relativos à pasta do app.py, então a
    cópia funciona em qualquer PC. A senha e as chaves da caixa postal ficam
    de fora: configure-as de novo no outro computador.
================================================================================
"""

from __future__ import annotations

import streamlit as st

from diagnostico_acustico.config import BASE_DIR, MIN_CLASSES_TREINO, MIN_SAMPLES_PER_CLASS, MODEL_PATH
from diagnostico_acustico.estilo import configurar_pagina
from diagnostico_acustico.pipeline import obter_status_base
from diagnostico_acustico.treino import carregar_modelo_salvo
from diagnostico_acustico.ui_acesso import render_acesso_admin
from diagnostico_acustico.ui_admin import render_tab_admin
from diagnostico_acustico.ui_central import (
    render_aba_caixa_entrada,
    render_aba_celulares,
    render_aba_textos,
    sincronizar_ao_entrar,
)
from diagnostico_acustico.ui_comum import _md, inicializar_sessao
from diagnostico_acustico.ui_diagnostico import render_tab_diagnostico
from diagnostico_acustico.ui_sidebar import render_sidebar


def main() -> None:
    configurar_pagina()
    inicializar_sessao()

    st.title("Diagnóstico Acústico de Falhas Mecânicas")
    st.caption("Análise multimodal (áudio + texto) para diagnóstico de falhas em equipamentos.")

    estado_modelo = carregar_modelo_salvo()
    status = obter_status_base(estado_modelo)

    if status.erro:
        st.error(f"Problema na base de conhecimento: {_md(status.erro)}")
    if estado_modelo is None:
        st.warning(
            "**Cold start** — ainda não existe um modelo treinado "
            f"(`{MODEL_PATH.relative_to(BASE_DIR).as_posix()}`). Use a aba **Alimentação (Admin)** para "
            f"cadastrar os primeiros casos conhecidos. Cada rótulo precisa acumular pelo menos "
            f"{MIN_SAMPLES_PER_CLASS} amostras para entrar no treino, e o primeiro treino exige "
            f"{MIN_CLASSES_TREINO} rótulos diferentes."
        )
    else:
        classes = estado_modelo.metadata.get("classes", [])
        st.caption(
            f"Modelo carregado — {len(classes)} classe(s) conhecida(s): {_md(', '.join(classes))}."
        )

    aba_admin, aba_diagnostico = st.tabs(["Alimentação (Admin)", "Diagnóstico (Operador)"])
    with aba_admin:
        if render_acesso_admin():
            sincronizar_ao_entrar(estado_modelo)
            # Títulos fixos: se mudassem (ex.: com o número de casos), o Streamlit
            # voltaria para a primeira aba a cada ação. O número aparece na barra lateral.
            alimentar, caixa, textos, celulares = st.tabs([
                "Alimentar", "Caixa de entrada", "Textos dos defeitos", "Celulares e sincronização",
            ])
            with alimentar:
                render_tab_admin(status, estado_modelo)
            with caixa:
                render_aba_caixa_entrada(estado_modelo, status)
            with textos:
                render_aba_textos(status)
            with celulares:
                render_aba_celulares(estado_modelo)
    with aba_diagnostico:
        render_tab_diagnostico(estado_modelo, status)

    render_sidebar(estado_modelo)  # por último: a checagem do ambiente não atrasa o conteúdo principal


if __name__ == "__main__":
    main()
