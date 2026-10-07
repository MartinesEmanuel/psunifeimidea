"""Orquestração usada pelas abas: alimentar, absorver, diagnosticar, excluir e status."""
from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from .audio import (
    AudioMetadata,
    carregar_amostras_audio,
    extrair_audio_de_video,
    obter_metadados_audio,
    preprocessar_audio_pipeline,
    validar_sinal_audio,
)
from .base_compartilhada import registrar_exclusoes
from .config import DATASET_PATH, Etapa, MAX_VIDEO_DURATION_WARNING_S, _sem_progresso, logger
from .dataset import (
    COLUNAS_META,
    COLUNAS_META_EXTRAS,
    adicionar_registro_dataset,
    carregar_ou_inicializar_dataset,
    gerar_uuid_curto,
    mapa_hashes,
    modelos_do_dataset,
    normalizar_modelo,
    normalizar_rotulo,
    pecas_registradas,
    remover_arquivos_caso,
    rotulos_do_dataset,
    salvar_arquivos_conhecidos,
    salvar_dataset_master,
)
from .erros import ModeloIndisponivelError, PipelineError
from .features import extrair_features_audio, montar_vetor_multimodal, validar_dimensao_vetor
from .recursos import _trava_processamento, obter_dimensao_vetor_texto
from .texto import (
    categorias_ignoradas_no_laudo,
    extrair_vetor_texto,
    limpar_texto,
    remover_termos_de_problema,
)
from .treino import (
    EstadoModelo,
    ResultadoTreino,
    _treinar_ou_adiar,
    assinatura_dados_treino,
    descrever_pendencias_treino,
    diagnosticar,
    pode_treinar,
    preparar_dados_treino,
    remover_artefatos_modelo,
)


# ==============================================================================
# 10. ORQUESTRAÇÃO DE ALTO NÍVEL (usada pelas duas abas da interface)
# ==============================================================================

@dataclass
class ResultadoExtracaoAudio:
    features_audio: np.ndarray
    wav_bytes: bytes
    sample_rate: int
    metadados: AudioMetadata
    ratio_fala: Optional[float]
    demucs_aplicado: bool
    avisos: List[str]
    hash_video: str


def calcular_hash_video(video_bytes: bytes) -> str:
    return hashlib.sha256(video_bytes).hexdigest()


def processar_video_e_extrair(video_bytes: bytes, nome_arquivo: str, etapa: Etapa = _sem_progresso) -> ResultadoExtracaoAudio:
    """Pipeline completo para UM único vídeo: extrai o áudio via ffmpeg +
    tempfile, valida o sinal, roda VAD/Demucs pontualmente e calcula as 56
    features acústicas."""
    sufixo = Path(nome_arquivo).suffix or ".mp4"
    etapa("extraindo o áudio (ffmpeg)", 0.02)
    caminho_wav = extrair_audio_de_video(video_bytes, sufixo)
    try:
        metadados = obter_metadados_audio(caminho_wav)  # Otimização 1
        y, sr = carregar_amostras_audio(caminho_wav)
    finally:
        caminho_wav.unlink(missing_ok=True)
    logger.info(
        "Áudio extraído de '%s': %.2fs @ %dHz (%d canal(is)).",
        nome_arquivo, metadados.duracao_s, metadados.sample_rate, metadados.canais,
    )

    avisos = validar_sinal_audio(y, sr)
    pre = preprocessar_audio_pipeline(y, sr, etapa)
    avisos += pre.avisos

    etapa("calculando as 56 características acústicas", 0.75)
    features_audio = extrair_features_audio(pre.audio, sr)

    buffer = io.BytesIO()
    sf.write(buffer, pre.audio, sr, format="WAV", subtype="PCM_16")

    return ResultadoExtracaoAudio(
        features_audio=features_audio, wav_bytes=buffer.getvalue(), sample_rate=sr,
        metadados=metadados, ratio_fala=pre.ratio_fala, demucs_aplicado=pre.demucs_aplicado,
        avisos=avisos, hash_video=calcular_hash_video(video_bytes),
    )


def registrar_caso_no_dataset(
    df_master: pd.DataFrame,
    rotulo: str,
    features_audio: np.ndarray,
    texto_laudo: str,
    audio_wav_bytes: bytes,
    dim_texto: int,
    extras: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, str, Tuple[Path, Path]]:
    """Gera o UUID, salva os arquivos conhecidos (áudio + texto) e insere uma
    linha no dataset mestre EM MEMÓRIA -- sem treinar. Isolar esta etapa
    permite acumular vários casos (upload em lote) e treinar apenas UMA vez
    no final, em vez de retreinar o LightGBM a cada item processado."""
    termos = categorias_ignoradas_no_laudo([*rotulos_do_dataset(df_master), rotulo])
    vetor_texto = extrair_vetor_texto(texto_laudo, termos)
    vetor_multimodal = montar_vetor_multimodal(features_audio, vetor_texto)
    validar_dimensao_vetor(vetor_multimodal, dim_texto)  # antes de gravar qualquer arquivo

    registro_id = gerar_uuid_curto(df_master)
    destino_audio, destino_texto = salvar_arquivos_conhecidos(rotulo, registro_id, audio_wav_bytes, texto_laudo)
    try:
        df_atualizado = adicionar_registro_dataset(
            df_master, registro_id, rotulo, destino_audio, destino_texto, vetor_multimodal, dim_texto,
            extras={"criado_em": datetime.now().isoformat(timespec="seconds"), **(extras or {})},
        )
    except Exception:
        remover_arquivos_caso(str(destino_audio), str(destino_texto))
        raise
    return df_atualizado, registro_id, (destino_audio, destino_texto)


def _persistir_novo_caso(
    df_master: pd.DataFrame,
    rotulo: str,
    features_audio: np.ndarray,
    texto_laudo: str,
    audio_wav_bytes: bytes,
    dim_texto: int,
    extras: Dict[str, str],
) -> Tuple[pd.DataFrame, str]:
    """Registra o caso E grava o CSV na hora. Se a gravação do CSV falhar, os
    arquivos recém-criados são apagados (nada de áudios órfãos)."""
    df_novo, registro_id, (audio, texto) = registrar_caso_no_dataset(
        df_master, rotulo, features_audio, texto_laudo, audio_wav_bytes, dim_texto, extras
    )
    try:
        salvar_dataset_master(df_novo)
    except Exception:
        remover_arquivos_caso(str(audio), str(texto))
        raise
    return df_novo, registro_id


@dataclass
class EntradaCaso:
    """Um bloco "Caso N" da aba Admin: um ou mais vídeos do MESMO problema,
    com o mesmo laudo e o mesmo rótulo."""
    caso_id: str
    numero: int
    arquivos: List[Any]  # UploadedFile(s) do Streamlit (ou qualquer objeto com .name e .getvalue())
    texto: str
    rotulo: str
    modelo: str = ""
    peca: str = ""


@dataclass
class ItemLote:
    caso_id: str
    numero_caso: int
    arquivo: str
    rotulo: str
    total_no_caso: int
    modelo: str = ""
    status: str = "pendente"  # "ok" | "duplicado" | "erro"
    registro_id: Optional[str] = None
    duplicado_de: Optional[str] = None
    erro: Optional[str] = None
    duracao_s: Optional[float] = None
    ratio_fala: Optional[float] = None
    demucs_aplicado: bool = False
    avisos: List[str] = field(default_factory=list)


@dataclass
class ResultadoLoteAdmin:
    itens: List[ItemLote]
    treinou: bool
    resultado_treino: Optional[ResultadoTreino] = None
    motivo_sem_treino: Optional[str] = None
    classes_aguardando: Dict[str, int] = field(default_factory=dict)
    erro_treino: Optional[str] = None

    @property
    def n_sucesso(self) -> int:
        return sum(1 for item in self.itens if item.status == "ok")

    @property
    def n_falha(self) -> int:
        return sum(1 for item in self.itens if item.status == "erro")

    def como_dict(self) -> Dict[str, Any]:
        """Versão "leve" (sem o modelo) para guardar no session_state."""
        rt = self.resultado_treino
        return {
            "itens": [asdict(item) for item in self.itens],
            "treinou": self.treinou,
            "classes_treinadas": rt.classes_treinadas if rt else [],
            "acuracia_cv": rt.acuracia_cv if rt else None,
            "n_folds_cv": rt.n_folds_cv if rt else 0,
            "n_amostras_treino": rt.n_amostras_treino if rt else 0,
            "motivo_sem_treino": self.motivo_sem_treino,
            "classes_aguardando": dict(self.classes_aguardando),
            "erro_treino": self.erro_treino,
            "interrompido": False,
        }


def validar_entradas_admin(entradas: Sequence[EntradaCaso]) -> List[str]:
    """Checa TODOS os blocos antes de começar o processamento (que pode ser
    demorado): é melhor avisar já que falta um rótulo do que depois de minutos."""
    problemas: List[str] = []
    for entrada in entradas:
        faltas = []
        if not entrada.arquivos:
            faltas.append("envie ao menos um vídeo")
        if not limpar_texto(entrada.texto):
            faltas.append("preencha o texto do laudo")
        if not limpar_texto(entrada.rotulo):
            faltas.append("informe o nome exato do problema (rótulo)")
        if not limpar_texto(entrada.modelo):
            faltas.append("informe o modelo da máquina")
        if faltas:
            problemas.append(f"Caso {entrada.numero}: {', '.join(faltas)}.")
    return problemas


def alimentar_sistema_em_lote(
    entradas: Sequence[EntradaCaso],
    progresso: Optional[Callable[[float, str], None]] = None,
    ao_processar_item: Optional[Callable[[ItemLote], None]] = None,
) -> ResultadoLoteAdmin:
    """ABA ADMIN: processa VÁRIOS vídeos de uma vez -- extrai o áudio e as
    features de cada um, registra cada caso no dataset_master.csv assim que
    fica pronto (nada se perde se o envio for interrompido) e, só no final,
    dispara UM ÚNICO retreino do LightGBM sobre o dataset atualizado."""
    progresso = progresso or (lambda _f, _t: None)
    tarefas = [(entrada, arquivo) for entrada in entradas for arquivo in entrada.arquivos]
    total = max(1, len(tarefas))

    with _trava_processamento():
        dim_texto = obter_dimensao_vetor_texto()
        df_master = carregar_ou_inicializar_dataset(dim_texto)
        hashes = mapa_hashes(df_master)

        itens: List[ItemLote] = []
        for k, (entrada, arquivo) in enumerate(tarefas):
            nome = str(getattr(arquivo, "name", f"arquivo_{k + 1}"))
            item = ItemLote(
                caso_id=entrada.caso_id, numero_caso=entrada.numero, arquivo=nome,
                rotulo=limpar_texto(entrada.rotulo), total_no_caso=len(entrada.arquivos),
                modelo=limpar_texto(entrada.modelo),
            )

            def etapa(texto: str, fracao: float, _k: int = k, _nome: str = nome) -> None:
                progresso((_k + fracao) / total, f"Vídeo {_k + 1} de {len(tarefas)} · {_nome} — {texto}...")

            try:
                if not (limpar_texto(entrada.texto) and item.rotulo and item.modelo):
                    raise PipelineError("o caso precisa de laudo, rótulo e modelo da máquina.")
                video_bytes = arquivo.getvalue()
                hash_video = calcular_hash_video(video_bytes)
                if hash_video in hashes:
                    item.status = "duplicado"
                    item.registro_id, item.rotulo = hashes[hash_video]
                    item.duplicado_de = item.registro_id
                else:
                    item.rotulo = normalizar_rotulo(item.rotulo, rotulos_do_dataset(df_master))
                    item.modelo = normalizar_modelo(item.modelo, modelos_do_dataset(df_master))
                    resultado_audio = processar_video_e_extrair(video_bytes, nome, etapa)
                    item.duracao_s = resultado_audio.metadados.duracao_s
                    item.ratio_fala = resultado_audio.ratio_fala
                    item.demucs_aplicado = resultado_audio.demucs_aplicado
                    item.avisos = list(resultado_audio.avisos)
                    if item.duracao_s > MAX_VIDEO_DURATION_WARNING_S:
                        item.avisos.append(
                            f"vídeo com {item.duracao_s / 60:.1f} min, acima dos "
                            f"{MAX_VIDEO_DURATION_WARNING_S // 60} min recomendados; considere cortar da próxima vez."
                        )
                    etapa("salvando na base de conhecimento", 0.95)
                    df_master, registro_id = _persistir_novo_caso(
                        df_master, item.rotulo, resultado_audio.features_audio, entrada.texto,
                        resultado_audio.wav_bytes, dim_texto,
                        extras={
                            "hash_video": hash_video, "arquivo_original": nome, "origem": "admin",
                            "modelo_maquina": item.modelo, "peca": limpar_texto(entrada.peca),
                        },
                    )
                    hashes[hash_video] = (registro_id, item.rotulo)
                    item.registro_id = registro_id
                    item.status = "ok"
            except PipelineError as exc:
                item.status, item.erro = "erro", str(exc)
            except Exception as exc:  # defesa extra: um item ruim não deve derrubar o lote inteiro
                logger.exception("Erro inesperado ao processar '%s' no lote de alimentação", nome)
                item.status, item.erro = "erro", f"erro inesperado: {exc}"
            itens.append(item)
            if ao_processar_item is not None:
                ao_processar_item(item)

        resultado = ResultadoLoteAdmin(itens=itens, treinou=False)
        if not any(item.status == "ok" for item in itens):
            resultado.motivo_sem_treino = "nenhum vídeo novo foi cadastrado neste envio."
            return resultado

        progresso(1.0, "Treinando a IA (LightGBM)...")
        try:
            treinou, resultado_treino, motivo, aguardando = _treinar_ou_adiar(df_master, dim_texto)
            resultado.treinou, resultado.resultado_treino = treinou, resultado_treino
            resultado.motivo_sem_treino, resultado.classes_aguardando = motivo, aguardando
        except PipelineError as exc:
            resultado.erro_treino = str(exc)
        return resultado


@dataclass
class ResultadoAbsorcao:
    registro_id: Optional[str]
    rotulo: str
    treinou: bool
    resultado_treino: Optional[ResultadoTreino] = None
    motivo_sem_treino: Optional[str] = None
    classes_aguardando: Dict[str, int] = field(default_factory=dict)
    duplicado_de: Optional[str] = None


def absorver_caso(
    rotulo: str,
    features_audio: np.ndarray,
    texto_laudo: str,
    audio_wav_bytes: bytes,
    hash_video: str = "",
    nome_arquivo: str = "",
    modelo_maquina: str = "",
    origem: str = "diagnostico",
    peca: str = "",
) -> ResultadoAbsorcao:
    """Passo final usado pela aba de Diagnóstico e pela caixa de entrada dos
    celulares: gera UUID, salva os arquivos conhecidos, insere no
    dataset_master.csv e RETREINA a IA respeitando a regra de acumulação
    (>= MIN_SAMPLES_PER_CLASS amostras por classe)."""
    with _trava_processamento():
        dim_texto = obter_dimensao_vetor_texto()
        df_master = carregar_ou_inicializar_dataset(dim_texto)
        rotulo_final = normalizar_rotulo(rotulo, rotulos_do_dataset(df_master))
        modelo_final = normalizar_modelo(modelo_maquina, modelos_do_dataset(df_master))

        hashes = mapa_hashes(df_master)
        if hash_video and hash_video in hashes:
            registro_existente, rotulo_existente = hashes[hash_video]
            return ResultadoAbsorcao(
                registro_id=registro_existente, rotulo=rotulo_existente, treinou=False,
                duplicado_de=registro_existente,
                motivo_sem_treino="este vídeo já estava cadastrado na base, então não foi adicionado de novo.",
            )

        df_master, registro_id = _persistir_novo_caso(
            df_master, rotulo_final, features_audio, texto_laudo, audio_wav_bytes, dim_texto,
            extras={
                "hash_video": hash_video, "arquivo_original": nome_arquivo, "origem": origem,
                "modelo_maquina": modelo_final, "peca": limpar_texto(peca),
            },
        )
        treinou, resultado_treino, motivo_sem_treino, classes_aguardando = _treinar_ou_adiar(df_master, dim_texto)
        return ResultadoAbsorcao(
            registro_id=registro_id, rotulo=rotulo_final, treinou=treinou,
            resultado_treino=resultado_treino, motivo_sem_treino=motivo_sem_treino,
            classes_aguardando=classes_aguardando,
        )


def gerar_diagnostico(
    video_bytes: bytes,
    nome_arquivo: str,
    texto_laudo: str,
    estado_modelo: Optional[EstadoModelo],
    etapa: Etapa = _sem_progresso,
    modelo_maquina: str = "",
) -> Dict[str, Any]:
    """ABA DIAGNÓSTICO -- Passo A: extrai features do vídeo/texto desconhecidos
    e chama predict_proba() do modelo carregado. Devolve as probabilidades e
    os dados brutos, para reaproveitar sem reprocessar o vídeo quando o
    operador confirmar (ou corrigir) o problema."""
    if estado_modelo is None:
        raise ModeloIndisponivelError(
            "ainda não existe um modelo treinado. Alimente o sistema pela aba 'Alimentação (Admin)' primeiro."
        )
    if not video_bytes:
        raise PipelineError("envie um vídeo desconhecido antes de gerar o diagnóstico.")
    if not limpar_texto(texto_laudo):
        raise PipelineError("o texto do laudo não pode estar vazio.")
    if not limpar_texto(modelo_maquina):
        raise PipelineError("informe o modelo da máquina.")

    with _trava_processamento():
        df_master = carregar_ou_inicializar_dataset(obter_dimensao_vetor_texto())
        modelo_final = normalizar_modelo(modelo_maquina, modelos_do_dataset(df_master))
        resultado_audio = processar_video_e_extrair(video_bytes, nome_arquivo, etapa)
        etapa("calculando as probabilidades", 0.95)
        termos = categorias_ignoradas_no_laudo([*rotulos_do_dataset(df_master), *estado_modelo.encoder.classes_])
        _, termos_retirados = remover_termos_de_problema(limpar_texto(texto_laudo), termos)
        vetor_texto = extrair_vetor_texto(texto_laudo, termos)
        vetor_multimodal = montar_vetor_multimodal(resultado_audio.features_audio, vetor_texto)
        probs = diagnosticar(vetor_multimodal, estado_modelo.modelo, estado_modelo.encoder)
        pecas = {rotulo: pecas_registradas(df_master, rotulo, modelo_final) for rotulo in probs}

    return {
        "probs": probs,
        "modelo_maquina": modelo_final,
        "pecas": pecas,
        "termos_retirados": sorted({t.casefold() for t in termos_retirados}),
        "features_audio": resultado_audio.features_audio.tolist(),
        "texto_laudo": texto_laudo,
        "wav_bytes": resultado_audio.wav_bytes,
        "hash_video": resultado_audio.hash_video,
        "nome_arquivo": nome_arquivo,
        "metadados": asdict(resultado_audio.metadados),
        "ratio_fala": resultado_audio.ratio_fala,
        "demucs_aplicado": resultado_audio.demucs_aplicado,
        "avisos": list(resultado_audio.avisos),
    }


def excluir_casos(uuids: Sequence[str]) -> Dict[str, Any]:
    """Remove casos do dataset (e seus arquivos de áudio/laudo) e retreina.
    Se depois da exclusão não houver dados suficientes, o modelo salvo é
    apagado -- ele conteria classes cujos exemplos não existem mais."""
    with _trava_processamento():
        dim_texto = obter_dimensao_vetor_texto()
        df_master = carregar_ou_inicializar_dataset(dim_texto)
        mascara = df_master["UUID"].isin(list(uuids))
        removidos = df_master[mascara]
        df_restante = df_master[~mascara].reset_index(drop=True)
        salvar_dataset_master(df_restante)
        for audio, texto in zip(removidos["audio_path"], removidos["text_path"]):
            remover_arquivos_caso(audio, texto)
        registrar_exclusoes(removidos)  # saem também dos outros computadores na próxima sincronização

        treinou, resultado_treino, motivo, _ = _treinar_ou_adiar(df_restante, dim_texto)
        if not treinou:
            remover_artefatos_modelo()
        return {
            "n_removidos": int(mascara.sum()),
            "treinou": treinou,
            "classes_treinadas": resultado_treino.classes_treinadas if resultado_treino else [],
            "motivo_sem_treino": motivo,
        }


def retreinar_agora() -> Tuple[bool, Optional[ResultadoTreino], Optional[str]]:
    """Treina a IA sobre o dataset atual (usado pelo botão "Treinar IA agora")."""
    with _trava_processamento():
        dim_texto = obter_dimensao_vetor_texto()
        df_master = carregar_ou_inicializar_dataset(dim_texto)
        treinou, resultado_treino, motivo, _ = _treinar_ou_adiar(df_master, dim_texto)
        return treinou, resultado_treino, motivo


@dataclass
class StatusBase:
    """Fotografia da base de conhecimento, usada para os painéis da interface."""
    total_casos: int = 0
    contagens: Dict[str, int] = field(default_factory=dict)
    contagens_modelos: Dict[str, int] = field(default_factory=dict)
    pode_treinar: bool = False
    modelo_desatualizado: bool = False
    motivo_sem_treino: Optional[str] = None
    tabela_casos: Optional[pd.DataFrame] = None
    erro: Optional[str] = None


@st.cache_data(show_spinner=False, max_entries=4)
def _ler_dataset_cacheado(assinatura: Tuple[int, int], dim_texto: int) -> pd.DataFrame:
    """Leitura do CSV reaproveitada entre cliques enquanto o arquivo não muda
    (a assinatura -- data de modificação e tamanho -- é a chave do cache)."""
    del assinatura
    return carregar_ou_inicializar_dataset(dim_texto)


def obter_status_base(estado_modelo: Optional[EstadoModelo]) -> StatusBase:
    try:
        dim_texto = obter_dimensao_vetor_texto()
        if DATASET_PATH.exists():
            info = DATASET_PATH.stat()
            df = _ler_dataset_cacheado((info.st_mtime_ns, info.st_size), dim_texto)
        else:
            df = carregar_ou_inicializar_dataset(dim_texto)
    except PipelineError as exc:
        return StatusBase(erro=str(exc))
    except Exception as exc:
        logger.exception("Falha ao ler o status da base")
        return StatusBase(erro=f"falha ao ler a base de conhecimento: {exc}")

    contagens = {str(k): int(v) for k, v in df["rotulo"].value_counts().items()} if not df.empty else {}
    modelos = df["modelo_maquina"].replace("", "Não informado") if not df.empty else pd.Series(dtype=str)
    df_treino, _ = preparar_dados_treino(df)
    treinavel = pode_treinar(df_treino)
    assinatura_modelo = estado_modelo.metadata.get("assinatura_dados") if estado_modelo else None
    desatualizado = treinavel and assinatura_modelo != assinatura_dados_treino(df_treino)
    return StatusBase(
        total_casos=len(df),
        contagens=contagens,
        contagens_modelos={str(k): int(v) for k, v in modelos.value_counts().items()},
        pode_treinar=treinavel,
        modelo_desatualizado=desatualizado,
        motivo_sem_treino=None if treinavel else descrever_pendencias_treino(contagens),
        tabela_casos=df[COLUNAS_META + COLUNAS_META_EXTRAS].copy(),
    )
