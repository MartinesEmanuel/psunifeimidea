"""Pacote para os celulares: o que o app precisa para diagnosticar sem internet.

O celular não consegue rodar o spaCy (o modelo de linguagem usado para ler o
laudo), então ele recebe uma versão da IA treinada SÓ com o som -- os mesmos
casos e as mesmas 56 características do site. A análise completa (som +
laudo) continua sendo feita aqui no computador, que devolve ao técnico uma
"segunda opinião" quando o caso chega pela caixa postal.

Conteúdo do pacote (.zip):
    manifesto.json          versão, categorias, métricas, contrato das características e hash de cada arquivo
    modelo_audio.json       a IA só de som (árvores do LightGBM, em JSON)
    textos_defeitos.json    causas, o que verificar e o que fazer de cada categoria
    pecas.json              peças cadastradas por categoria e modelo de máquina
    modelos_maquina.json    modelos de máquina já cadastrados (para sugerir ao digitar)
    kit_conformidade/       áudios de teste + o resultado esperado: o app confere se
                            calcula as características exatamente como o site
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from .config import (
    FFMPEG_TARGET_SR,
    FORMATO_PACOTE,
    HOP_SEC,
    LGBM_PARAMS,
    N_MFCC,
    N_PEAKS,
    PUBLICACOES_DIR,
    ROTULO_REFERENCIA,
    WINDOW_SEC,
    logger,
)
from .conhecimento import carregar_textos
from .dataset import _chave_modelo, _chave_rotulo, _gravar_atomico, carregar_ou_inicializar_dataset, modelos_do_dataset
from .erros import PipelineError
from .features import AUDIO_FEATURE_COLUMNS, extrair_features_audio
from .recursos import _trava_processamento, obter_dimensao_vetor_texto
from .texto import limpar_texto
from .treino import avaliar_por_validacao_cruzada, pode_treinar, preparar_dados_treino

LISTA_PUBLICACOES_PATH = PUBLICACOES_DIR / "publicacoes.json"

# Como o app deve calcular as 56 características para dar o mesmo resultado
# do site. O kit de conformidade é a prova: o app calcula as características
# dos áudios do kit e compara com o esperado.json.
CONTRATO_CARACTERISTICAS: Dict[str, Any] = {
    "versao": 1,
    "audio": {"taxa_amostragem_hz": FFMPEG_TARGET_SR, "canais": 1, "amostras": "float32 entre -1 e 1 (WAV PCM 16 bits / 32768)"},
    "janelas": {
        "tamanho_s": WINDOW_SEC, "passo_s": HOP_SEC,
        "regra": "janelas de 2 s com passo de 1 s enquanto couberem; se o áudio tiver menos de 2 s, uma única janela "
                 "com o áudio inteiro. O vetor final é a MÉDIA de cada característica entre as janelas.",
    },
    "colunas": list(AUDIO_FEATURE_COLUMNS),
    "por_janela": {
        "rms": "sqrt(média(y²)) + 1e-9",
        "crest_factor": "max(|y|) / rms",
        "zcr": "média de librosa.feature.zero_crossing_rate(y) (quadro 2048, passo 512, center=True)",
        "kurtosis": "scipy.stats.kurtosis(y) (Fisher, bias=True)",
        "skewness": "scipy.stats.skew(y) (bias=True)",
        "spec_*": "média de librosa.feature.spectral_centroid / spectral_bandwidth / spectral_rolloff (0,85) / "
                  "spectral_flatness, com n_fft=2048, passo 512, janela Hann, center=True (padrões da librosa)",
        "mfcc": f"librosa.feature.mfcc(n_mfcc={N_MFCC}) com os padrões da librosa (128 filtros mel Slaney, "
                "power_to_db ref=1, top_db=80, DCT tipo 2 orto); média e desvio-padrão de cada coeficiente",
        "peak_freq_1..5_hz": f"scipy.signal.find_peaks(|rfft(y)|, distance=5); os {N_PEAKS} picos mais altos, "
                             "em ordem crescente de frequência; completar com 0",
        "envelope_freq_hz / envelope_strength": "e = |hilbert(y)| - média; E = |rfft(e)| sem o termo DC; "
                                                "frequência do maior pico de E e força = pico / soma(E)",
    },
}

COMO_CALCULAR_PROBABILIDADES = (
    "Para cada árvore de modelo['tree_info'], descer pelos nós: em 'split_feature' compare a característica "
    "com 'threshold' (decision_type '<=': valor <= limiar vai para left_child, senão right_child) até chegar a "
    "uma folha ('leaf_value'). Some as folhas. Binário (num_class = 1): p = 1/(1+e^-soma) é a probabilidade da "
    "SEGUNDA categoria de 'classes'. Multiclasse: a árvore i pertence à categoria i % num_class; aplique softmax."
)


# ------------------------------------------------------------ lista de publicações

def listar_publicacoes() -> List[Dict[str, Any]]:
    if not LISTA_PUBLICACOES_PATH.exists():
        return []
    try:
        return list(json.loads(LISTA_PUBLICACOES_PATH.read_text(encoding="utf-8")))
    except Exception:
        logger.exception("Falha ao ler a lista de publicações")
        return []


def _salvar_publicacoes(lista: List[Dict[str, Any]]) -> None:
    PUBLICACOES_DIR.mkdir(parents=True, exist_ok=True)
    _gravar_atomico(
        LISTA_PUBLICACOES_PATH,
        lambda caminho: caminho.write_text(json.dumps(lista, ensure_ascii=False, indent=2), encoding="utf-8"),
    )


def publicacoes_pendentes() -> List[Dict[str, Any]]:
    """Pacotes gerados que ainda não chegaram à caixa postal."""
    return [p for p in listar_publicacoes() if not p.get("enviado_em")]


def marcar_publicacao_enviada(versao: int) -> None:
    lista = listar_publicacoes()
    for p in lista:
        if int(p["versao"]) == int(versao):
            p["enviado_em"] = datetime.now().isoformat(timespec="seconds")
    _salvar_publicacoes(lista)


def caminho_pacote(publicacao: Dict[str, Any]):
    return PUBLICACOES_DIR / publicacao["arquivo"]


# ------------------------------------------------------------------- conteúdo

def _sem_infinitos(valor: Any) -> Any:
    """O JSON padrão (e o do celular) não aceita Infinity/NaN; o LightGBM às
    vezes usa limiares infinitos."""
    if isinstance(valor, float) and not math.isfinite(valor):
        return 1e308 if valor > 0 else -1e308 if valor < 0 else 0.0
    if isinstance(valor, dict):
        return {k: _sem_infinitos(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_sem_infinitos(v) for v in valor]
    return valor


def treinar_modelo_audio(df_treino) -> Tuple[LGBMClassifier, LabelEncoder, Optional[float], int]:
    """A mesma IA do site, mas só com as 56 características de som."""
    X = np.nan_to_num(df_treino[list(AUDIO_FEATURE_COLUMNS)].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_treino["rotulo"].astype(str).to_numpy())
    try:
        acuracia, n_folds = avaliar_por_validacao_cruzada(X, y)
    except Exception:
        logger.exception("Falha na validação cruzada do modelo só de som")
        acuracia, n_folds = None, 0
    modelo = LGBMClassifier(**LGBM_PARAMS)
    modelo.fit(X, y)
    return modelo, encoder, acuracia, n_folds


def _sinais_do_kit() -> List[Tuple[str, np.ndarray]]:
    """Três sinais sintéticos e sempre iguais (semente fixa), com características bem diferentes."""
    sr = FFMPEG_TARGET_SR
    t = np.arange(int(5.0 * sr)) / sr
    rng = np.random.default_rng(2026)
    tom = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.02 * rng.standard_normal(t.size)
    impactos = 0.25 * np.sin(2 * np.pi * 900 * t) + 0.02 * rng.standard_normal(t.size)
    for inicio in range(0, t.size, int(0.05 * sr)):  # batida a cada 50 ms, decaindo
        fim = min(t.size, inicio + 400)
        impactos[inicio:fim] += 0.6 * np.exp(-np.arange(fim - inicio) / 60.0)
    ruido = np.convolve(rng.standard_normal(t.size), np.ones(8) / 8, mode="same") * 0.2
    ruido += 0.1 * np.sin(2 * np.pi * 60 * t)
    return [
        ("kit_tom_120hz.wav", tom),
        ("kit_impactos_900hz.wav", impactos),
        ("kit_ruido_60hz.wav", ruido),
    ]


def _kit_conformidade(modelo: LGBMClassifier, encoder: LabelEncoder) -> Dict[str, bytes]:
    arquivos: Dict[str, bytes] = {}
    esperado: Dict[str, Any] = {}
    for nome, sinal in _sinais_do_kit():
        buffer = io.BytesIO()
        sf.write(buffer, np.clip(sinal, -1.0, 1.0), FFMPEG_TARGET_SR, format="WAV", subtype="PCM_16")
        wav = buffer.getvalue()
        # Lê de volta do WAV (16 bits), exatamente como o celular vai ler.
        y, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
        caracteristicas = extrair_features_audio(y, sr)
        probs = modelo.predict_proba(caracteristicas.reshape(1, -1))[0]
        arquivos[f"kit_conformidade/{nome}"] = wav
        esperado[nome] = {
            "caracteristicas": dict(zip(AUDIO_FEATURE_COLUMNS, map(float, caracteristicas))),
            "probabilidades": {str(c): float(p) for c, p in zip(encoder.classes_, probs)},
        }
    arquivos["kit_conformidade/esperado.json"] = json.dumps(
        {"tolerancia_relativa_sugerida": 1e-3, "audios": _sem_infinitos(esperado)}, ensure_ascii=False, indent=2
    ).encode("utf-8")
    return arquivos


def _pecas_para_celular(df) -> List[Dict[str, Any]]:
    """Peças da Alimentação agrupadas por categoria + modelo de máquina + peça."""
    if df.empty or "peca" not in df.columns:
        return []
    grupos: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rotulo, modelo, peca in zip(df["rotulo"], df["modelo_maquina"], df["peca"]):
        peca = limpar_texto(peca)
        if not peca:
            continue
        chave = (str(rotulo), _chave_modelo(modelo), _chave_rotulo(peca))
        item = grupos.setdefault(chave, {"categoria": str(rotulo), "modelo_maquina": limpar_texto(modelo),
                                         "peca": peca, "casos": 0})
        item["casos"] += 1
    return sorted(grupos.values(), key=lambda i: (i["categoria"], i["modelo_maquina"], -i["casos"], i["peca"]))


def gerar_pacote(notas: str = "") -> Dict[str, Any]:
    """Treina a IA só de som com os casos atuais, monta o .zip e o registra
    como publicação pendente (vai para a caixa postal na próxima sincronização)."""
    with _trava_processamento():
        df = carregar_ou_inicializar_dataset(obter_dimensao_vetor_texto())
        df_treino, _ = preparar_dados_treino(df)
        if not pode_treinar(df_treino):
            raise PipelineError(
                "ainda não há casos suficientes para treinar a IA, então não há o que enviar aos celulares."
            )
        modelo, encoder, acuracia, n_folds = treinar_modelo_audio(df_treino)
        classes = [str(c) for c in encoder.classes_]

        conteudo: Dict[str, bytes] = {
            "modelo_audio.json": json.dumps(_sem_infinitos({
                "formato": "lightgbm-dump",
                "classes": classes,
                "colunas": list(AUDIO_FEATURE_COLUMNS),
                "como_calcular": COMO_CALCULAR_PROBABILIDADES,
                "modelo": modelo.booster_.dump_model(),
            }), ensure_ascii=False).encode("utf-8"),
            "textos_defeitos.json": json.dumps(carregar_textos(), ensure_ascii=False, indent=2).encode("utf-8"),
            "pecas.json": json.dumps(_pecas_para_celular(df), ensure_ascii=False, indent=2).encode("utf-8"),
            "modelos_maquina.json": json.dumps(modelos_do_dataset(df), ensure_ascii=False, indent=2).encode("utf-8"),
            **_kit_conformidade(modelo, encoder),
        }

        lista = listar_publicacoes()
        versao = max((int(p["versao"]) for p in lista), default=0) + 1
        criado_em = datetime.now().isoformat(timespec="seconds")
        manifesto = {
            "formato": FORMATO_PACOTE,
            "versao": versao,
            "criado_em": criado_em,
            "notas": limpar_texto(notas),
            "categorias": classes,
            "categoria_referencia": ROTULO_REFERENCIA,
            "metricas": {
                "casos_no_treino": int(len(df_treino)),
                "casos_por_categoria": {str(k): int(v) for k, v in df_treino["rotulo"].value_counts().items()},
                "acerto_estimado_so_som": acuracia,
                "partes_validacao_cruzada": n_folds,
            },
            "contrato_caracteristicas": CONTRATO_CARACTERISTICAS,
            "arquivos": {nome: hashlib.sha256(dados).hexdigest() for nome, dados in sorted(conteudo.items())},
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifesto.json", json.dumps(manifesto, ensure_ascii=False, indent=2))
            for nome, dados in sorted(conteudo.items()):
                zf.writestr(nome, dados)
        dados_zip = buffer.getvalue()

        PUBLICACOES_DIR.mkdir(parents=True, exist_ok=True)
        arquivo = f"pacote_v{versao:04d}.zip"
        _gravar_atomico(PUBLICACOES_DIR / arquivo, lambda caminho: caminho.write_bytes(dados_zip))
        publicacao = {
            "versao": versao, "criado_em": criado_em, "arquivo": arquivo,
            "sha256": hashlib.sha256(dados_zip).hexdigest(), "tamanho": len(dados_zip),
            "notas": manifesto["notas"], "categorias": classes, "acerto_estimado_so_som": acuracia,
            "enviado_em": None,
        }
        _salvar_publicacoes(lista + [publicacao])
        logger.info("Pacote v%d gerado: %d bytes, categorias=%s", versao, len(dados_zip), classes)
        return publicacao
