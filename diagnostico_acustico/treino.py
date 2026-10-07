"""Treino do LightGBM, modelo salvo e diagnóstico."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .config import (
    ENCODER_PATH,
    LGBM_PARAMS,
    METADATA_PATH,
    MIN_CLASSES_TREINO,
    MIN_SAMPLES_PER_CLASS,
    MODELOS_DIR,
    MODEL_PATH,
    N_FOLDS_MAX_CV,
    ROTULO_REFERENCIA,
    logger,
)
from .dataset import _gravar_atomico, _resolver_caminho, eh_referencia, rotulos_do_dataset
from .erros import DatasetError, ExtracaoFeaturesError, ModeloIndisponivelError, PipelineError
from .features import AUDIO_FEATURE_COLUMNS, nomes_colunas_features
from .texto import categorias_ignoradas_no_laudo, extrair_vetor_texto


# ==============================================================================
# 9. TREINO DO MODELO (LightGBM -- Cold Start / Regra de Acumulação)
# ==============================================================================

@dataclass
class ResultadoTreino:
    modelo: LGBMClassifier
    encoder: LabelEncoder
    classes_treinadas: List[str]
    classes_aguardando: Dict[str, int]
    n_amostras_treino: int
    acuracia_cv: Optional[float] = None
    n_folds_cv: int = 0


@dataclass
class EstadoModelo:
    modelo: LGBMClassifier
    encoder: LabelEncoder
    metadata: Dict[str, Any]


def preparar_dados_treino(df_master: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """REGRA DE ACUMULAÇÃO (Seção 3): agrupa por rótulo e mantém no treino
    apenas as classes com >= MIN_SAMPLES_PER_CLASS amostras. As demais linhas
    permanecem intactas no CSV, apenas ficam de fora do LabelEncoder/LightGBM
    até acumularem amostras suficientes."""
    if df_master.empty:
        return df_master.copy(), {}
    contagens = df_master["rotulo"].value_counts()
    classes_validas = contagens[contagens >= MIN_SAMPLES_PER_CLASS].index.tolist()
    classes_aguardando = {str(k): int(v) for k, v in contagens[contagens < MIN_SAMPLES_PER_CLASS].items()}
    df_treino = df_master[df_master["rotulo"].isin(classes_validas)].copy()
    return df_treino, classes_aguardando


def assinatura_dados_treino(df_treino: pd.DataFrame) -> str:
    """Identifica o conjunto exato de casos usado num treino (para saber se o
    modelo salvo está desatualizado em relação ao dataset)."""
    ids = sorted(str(u) for u in df_treino["UUID"]) if not df_treino.empty else []
    return hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()


def pode_treinar(df_treino: pd.DataFrame) -> bool:
    return not df_treino.empty and df_treino["rotulo"].nunique() >= MIN_CLASSES_TREINO


def descrever_pendencias_treino(contagens: Dict[str, int]) -> str:
    """Explica, em linguagem direta, o que falta para o primeiro treino."""
    if not contagens:
        return (
            f"ainda não há casos cadastrados. O primeiro treino exige pelo menos "
            f"{MIN_CLASSES_TREINO} problemas diferentes com {MIN_SAMPLES_PER_CLASS} casos cada. "
            f"Os vídeos de {ROTULO_REFERENCIA} (máquina funcionando normal) contam como uma dessas categorias."
        )
    prontas = sorted(r for r, n in contagens.items() if n >= MIN_SAMPLES_PER_CLASS)
    faltando = sorted(
        ((r, MIN_SAMPLES_PER_CLASS - n) for r, n in contagens.items() if n < MIN_SAMPLES_PER_CLASS),
        key=lambda kv: (kv[1], kv[0]),
    )
    partes = [
        f"o primeiro treino exige pelo menos {MIN_CLASSES_TREINO} problemas diferentes "
        f"com {MIN_SAMPLES_PER_CLASS} casos cada."
    ]
    if prontas:
        partes.append(f"Já prontos: {', '.join(prontas)}.")
    precisa = MIN_CLASSES_TREINO - len(prontas)
    if faltando:
        partes.append(
            "Faltam: " + "; ".join(f"{r} (+{n} caso{'s' if n > 1 else ''})" for r, n in faltando[:6])
            + ("..." if len(faltando) > 6 else "") + "."
        )
    if len(faltando) < precisa:
        novos = precisa - len(faltando)
        partes.append(f"Cadastre também {novos} problema{'s' if novos > 1 else ''} diferente{'s' if novos > 1 else ''}.")
    if not any(eh_referencia(r) for r in contagens):
        partes.append(
            f"Os vídeos de {ROTULO_REFERENCIA} (máquina funcionando normal) contam como uma dessas categorias."
        )
    return " ".join(partes)


def avaliar_por_validacao_cruzada(X: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], int]:
    """Estima a acurácia em dados NÃO vistos no treino (validação cruzada
    estratificada). Não muda o modelo final -- só informa o quão confiável
    ele tende a ser. Com poucos casos por classe a estimativa é grosseira."""
    menor_classe = int(np.bincount(y).min())
    n_folds = min(N_FOLDS_MAX_CV, menor_classe)
    if n_folds < 2:
        return None, 0
    acertos = 0
    for idx_treino, idx_teste in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X, y):
        modelo = LGBMClassifier(**LGBM_PARAMS)
        modelo.fit(X[idx_treino], y[idx_treino])
        acertos += int((modelo.predict(X[idx_teste]) == y[idx_teste]).sum())
    return acertos / len(y), n_folds


def _dump_atomico(objeto: Any, destino: Path) -> None:
    _gravar_atomico(destino, lambda caminho: joblib.dump(objeto, caminho))


def _vetores_texto_atualizados(df_treino: pd.DataFrame, termos: Sequence[str], dim_texto: int) -> np.ndarray:
    """Recalcula o vetor de texto de cada caso a partir do laudo salvo, sem os
    nomes das categorias cadastradas. Necessário porque essa lista cresce a
    cada categoria nova: um laudo antigo pode conter o nome de uma categoria
    que ainda não existia quando ele foi salvo. Se o arquivo do laudo não
    existir mais, mantém o vetor gravado no CSV."""
    colunas_texto = nomes_colunas_features(dim_texto)[len(AUDIO_FEATURE_COLUMNS):]
    vetores = df_treino[colunas_texto].to_numpy(dtype=np.float64, copy=True)  # cópia gravável (pandas 3)
    for i, caminho in enumerate(df_treino["text_path"]):
        arquivo = _resolver_caminho(caminho) if str(caminho).strip() else None
        if arquivo is not None and arquivo.is_file():
            vetores[i] = extrair_vetor_texto(arquivo.read_text(encoding="utf-8"), termos)
    return vetores


def treinar_modelo(
    df_treino: pd.DataFrame, dim_texto: int, classes_aguardando: Dict[str, int], termos: Sequence[str]
) -> ResultadoTreino:
    """Treina o LightGBM com hiperparâmetros FIXOS (sem GridSearchCV) e
    sobrescreve modelo_lgbm.joblib, o LabelEncoder e os metadados em disco."""
    if df_treino.empty:
        raise DatasetError("não há amostras suficientes para treinar (nenhuma classe atingiu o mínimo exigido).")

    colunas_features = nomes_colunas_features(dim_texto)
    faltantes = [c for c in colunas_features if c not in df_treino.columns]
    if faltantes:
        raise ExtracaoFeaturesError(f"colunas de features ausentes no dataset de treino: {faltantes[:5]}")

    X_audio = df_treino[list(AUDIO_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    X = np.hstack([X_audio, _vetores_texto_atualizados(df_treino, termos, dim_texto)])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df_treino["rotulo"].astype(str).to_numpy())

    try:
        acuracia_cv, n_folds = avaliar_por_validacao_cruzada(X, y)
    except Exception:
        logger.exception("Falha na validação cruzada (o treino segue normalmente)")
        acuracia_cv, n_folds = None, 0

    try:
        modelo = LGBMClassifier(**LGBM_PARAMS)
        modelo.fit(X, y)  # treino ultra-rápido: um único .fit(), sem busca de hiperparâmetros
    except Exception as exc:
        raise PipelineError(f"falha ao treinar o LightGBM: {exc}") from exc

    contagem = df_treino["rotulo"].astype(str).value_counts()
    metadata = {
        "colunas_features": colunas_features,
        "dim_texto": dim_texto,
        "classes": encoder.classes_.tolist(),
        "n_amostras_treino": int(len(df_treino)),
        "contagem_por_classe": {str(k): int(v) for k, v in contagem.items()},
        "acuracia_cv": acuracia_cv,
        "n_folds_cv": n_folds,
        "assinatura_dados": assinatura_dados_treino(df_treino),
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        MODELOS_DIR.mkdir(parents=True, exist_ok=True)
        _dump_atomico(modelo, MODEL_PATH)
        _dump_atomico(encoder, ENCODER_PATH)
        _gravar_atomico(
            METADATA_PATH,
            lambda caminho: caminho.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"),
        )
    except Exception as exc:
        raise PipelineError(f"modelo treinado, mas falhou ao salvar em disco: {exc}") from exc

    logger.info("LightGBM treinado: %d amostras, classes=%s, acurácia CV=%s", len(df_treino), metadata["classes"], acuracia_cv)
    return ResultadoTreino(
        modelo=modelo,
        encoder=encoder,
        classes_treinadas=encoder.classes_.tolist(),
        classes_aguardando=classes_aguardando,
        n_amostras_treino=len(df_treino),
        acuracia_cv=acuracia_cv,
        n_folds_cv=n_folds,
    )


def remover_artefatos_modelo() -> None:
    for caminho in (MODEL_PATH, ENCODER_PATH, METADATA_PATH):
        caminho.unlink(missing_ok=True)


def _assinatura_arquivos_modelo() -> Optional[Tuple[int, ...]]:
    caminhos = (MODEL_PATH, ENCODER_PATH, METADATA_PATH)
    if not all(c.exists() for c in caminhos):
        return None
    return tuple(c.stat().st_mtime_ns for c in caminhos)


@st.cache_resource(show_spinner=False, max_entries=2)
def _carregar_modelo_do_disco(assinatura: Tuple[int, ...]) -> EstadoModelo:
    """Só relê os arquivos quando eles mudam (a assinatura são as datas de
    modificação), em vez de a cada clique na interface. Atenção: o parâmetro
    NÃO pode começar com "_", senão o Streamlit o ignora na chave do cache."""
    del assinatura  # usada apenas como chave do cache
    modelo = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return EstadoModelo(modelo=modelo, encoder=encoder, metadata=metadata)


def carregar_modelo_salvo() -> Optional[EstadoModelo]:
    """COLD START: se os artefatos ainda não existirem em disco, retorna None
    (sem levantar exceção) -- a interface apenas avisa que precisa de dados."""
    assinatura = _assinatura_arquivos_modelo()
    if assinatura is None:
        return None
    try:
        return _carregar_modelo_do_disco(assinatura)
    except Exception as exc:
        logger.error("Falha ao carregar o modelo salvo em disco: %s", exc)
        return None


def diagnosticar(vetor_multimodal: np.ndarray, modelo: LGBMClassifier, encoder: LabelEncoder) -> Dict[str, float]:
    """Roda predict_proba() do modelo carregado em memória sobre o vetor multimodal."""
    X = vetor_multimodal.reshape(1, -1)
    n_esperado = getattr(modelo, "n_features_in_", X.shape[1])
    if X.shape[1] != n_esperado:
        raise ModeloIndisponivelError(
            f"o modelo salvo espera {n_esperado} atributos, mas o vetor atual tem {X.shape[1]}. "
            "Retreine a IA pela aba de Alimentação."
        )
    try:
        probs = modelo.predict_proba(X)[0]
    except Exception as exc:
        raise PipelineError(f"falha ao gerar o diagnóstico: {exc}") from exc
    return {str(rotulo): float(p) for rotulo, p in zip(encoder.classes_, probs)}


def _treinar_ou_adiar(df_master: pd.DataFrame, dim_texto: int) -> Tuple[bool, Optional[ResultadoTreino], Optional[str], Dict[str, int]]:
    """Aplica a regra de acumulação sobre o dataset já atualizado e treina o
    LightGBM se houver pelo menos 2 classes com amostras suficientes."""
    df_treino, classes_aguardando = preparar_dados_treino(df_master)
    if not pode_treinar(df_treino):
        contagens = {str(k): int(v) for k, v in df_master["rotulo"].value_counts().items()} if not df_master.empty else {}
        return False, None, descrever_pendencias_treino(contagens), classes_aguardando
    termos = categorias_ignoradas_no_laudo(rotulos_do_dataset(df_master))
    resultado_treino = treinar_modelo(df_treino, dim_texto, classes_aguardando, termos)
    return True, resultado_treino, None, classes_aguardando
