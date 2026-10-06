"""As 56 características acústicas e o vetor multimodal (áudio + texto)."""
from __future__ import annotations

from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert
from scipy.stats import kurtosis, skew

from .config import HOP_SEC, N_MFCC, N_PEAKS, WINDOW_SEC
from .erros import ExtracaoFeaturesError, PipelineError


# ==============================================================================
# 5. EXTRAÇÃO DE FEATURES ACÚSTICAS (56 parâmetros)
# ==============================================================================

AUDIO_FEATURE_COLUMNS: List[str] = (
    ["rms", "crest_factor", "zcr", "kurtosis", "skewness",
     "spec_centroid", "spec_bandwidth", "spec_rolloff", "spec_flatness"]
    + [f"mfcc{i + 1}_{stat}" for i in range(N_MFCC) for stat in ("mean", "std")]
    + [f"peak_freq_{i + 1}_hz" for i in range(N_PEAKS)]
    + ["envelope_freq_hz", "envelope_strength"]
)
assert len(AUDIO_FEATURE_COLUMNS) == 56, "A extração deve gerar exatamente 56 parâmetros acústicos."


def _extrair_parametros_janela(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extrai os 56 parâmetros acústicos de uma única janela de áudio."""
    eps = 1e-9

    # --- Domínio do tempo ---
    rms = float(np.sqrt(np.mean(np.square(y))) + eps)
    crest_factor = float(np.max(np.abs(y)) / rms) if rms > 0 else 0.0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    kurt = float(kurtosis(y))
    skw = float(skew(y))

    # --- Domínio da frequência (espectrais) ---
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # --- MFCCs (timbre) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_feats: Dict[str, float] = {}
    for i in range(N_MFCC):
        mfcc_feats[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
        mfcc_feats[f"mfcc{i + 1}_std"] = float(np.std(mfcc[i]))

    # --- Picos de frequência dominantes (top 5) ---
    fft_vals = np.abs(np.fft.rfft(y))
    fft_freqs = np.fft.rfftfreq(len(y), 1 / sr)
    peaks_idx, _ = find_peaks(fft_vals, distance=5)
    if len(peaks_idx) > 0:
        top_peaks = peaks_idx[np.argsort(fft_vals[peaks_idx])[::-1][:N_PEAKS]]
        peak_freqs = sorted(fft_freqs[top_peaks].tolist())
    else:
        peak_freqs = []
    peak_freqs = (peak_freqs + [0.0] * N_PEAKS)[:N_PEAKS]

    # --- Envelope (modulação de amplitude) ---
    envelope = np.abs(hilbert(y))
    env_centered = envelope - np.mean(envelope)
    env_fft = np.abs(np.fft.rfft(env_centered))
    env_freqs = np.fft.rfftfreq(len(env_centered), 1 / sr)
    if len(env_fft) > 1:
        env_fft_no_dc = env_fft[1:]
        env_freqs_no_dc = env_freqs[1:]
        idx_max = int(np.argmax(env_fft_no_dc))
        envelope_freq_hz = float(env_freqs_no_dc[idx_max])
        envelope_strength = float(env_fft_no_dc[idx_max] / (np.sum(env_fft_no_dc) + eps))
    else:
        envelope_freq_hz, envelope_strength = 0.0, 0.0

    return {
        "rms": rms, "crest_factor": crest_factor, "zcr": zcr,
        "kurtosis": kurt, "skewness": skw,
        "spec_centroid": spec_centroid, "spec_bandwidth": spec_bandwidth,
        "spec_rolloff": spec_rolloff, "spec_flatness": spec_flatness,
        **mfcc_feats,
        "peak_freq_1_hz": peak_freqs[0], "peak_freq_2_hz": peak_freqs[1],
        "peak_freq_3_hz": peak_freqs[2], "peak_freq_4_hz": peak_freqs[3],
        "peak_freq_5_hz": peak_freqs[4],
        "envelope_freq_hz": envelope_freq_hz, "envelope_strength": envelope_strength,
    }


def extrair_features_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Extrai o vetor de 56 features acústicas, calculado como a média das
    janelas deslizantes (WINDOW_SEC/HOP_SEC) sobre o áudio inteiro."""
    try:
        duracao = len(y) / sr if sr else 0.0

        janelas: List[Dict[str, float]] = []
        t = 0.0
        while t + WINDOW_SEC <= duracao:
            inicio, fim = int(t * sr), int((t + WINDOW_SEC) * sr)
            janelas.append(_extrair_parametros_janela(y[inicio:fim], sr))
            t += HOP_SEC
        if not janelas:
            janelas = [_extrair_parametros_janela(y, sr)]

        df_janelas = pd.DataFrame(janelas)[AUDIO_FEATURE_COLUMNS]
        vetor_medio = df_janelas.mean(axis=0).to_numpy(dtype=np.float64)
        return np.nan_to_num(vetor_medio, nan=0.0, posinf=0.0, neginf=0.0)
    except PipelineError:
        raise
    except Exception as exc:
        raise ExtracaoFeaturesError(f"falha ao extrair features acústicas: {exc}") from exc




# ==============================================================================
# 7. FUSÃO MULTIMODAL (Áudio + Texto)
# ==============================================================================

def nomes_colunas_features(dim_texto: int) -> List[str]:
    """Nomes de coluna fixos para o vetor multimodal: 56 features de áudio +
    `dim_texto` dimensões do vetor textual do spaCy."""
    return list(AUDIO_FEATURE_COLUMNS) + [f"texto_vec_{i:03d}" for i in range(dim_texto)]


def montar_vetor_multimodal(features_audio: np.ndarray, vetor_texto: np.ndarray) -> np.ndarray:
    """
    ==== FUSÃO MULTIMODAL POR CONCATENAÇÃO (Áudio + Texto) ====

    Esta é a etapa central da multimodalidade do sistema (no código original
    chamada de "late fusion"; o termo acadêmico mais preciso é fusão em nível
    de atributos, ou *feature-level fusion*, porque existe um único
    classificador depois da concatenação):

        vetor_final = [ Audio_Features_56 | Text_Vector_D ]

    Por que essa abordagem é a correta para este problema:

      1. Tamanho fixo e determinístico: 56 (áudio) + D (texto, fixo pelo
         modelo spaCy carregado) = 56 + D colunas SEMPRE, não importa o
         conteúdo do texto ou a duração do áudio de entrada. É exatamente
         isso que permite retreinar o LightGBM incrementalmente (a cada novo
         caso absorvido) sem NENHUM risco de "shape mismatch" -- o problema
         clássico de usar TF-IDF em online learning, onde o vocabulário (e,
         portanto, o número de colunas de entrada) cresce a cada palavra
         inédita vista pelo sistema.

      2. Cada modalidade preserva sua própria semântica até chegar ao
         classificador: as 56 primeiras posições carregam a "assinatura
         vibracional/acústica" da falha mecânica (RMS, MFCCs, picos
         espectrais, envelope de amplitude etc.); as posições seguintes
         carregam o "contexto textual" do laudo (sintomas relatados pelo
         operador, palavras-chave do problema). O LightGBM, por ser baseado
         em árvores de decisão, aprende sozinho -- através dos splits em cada
         nó -- QUAIS dimensões (de qualquer uma das duas modalidades) são
         mais discriminativas para cada classe de defeito. Não é necessário
         nenhum peso manual entre as modalidades.

      3. Robustez dimensional: qualquer inconsistência de tamanho entre o
         vetor calculado e as colunas esperadas do dataset_master.csv é
         detectada e barrada ANTES de gravar qualquer arquivo (ver
         `validar_dimensao_vetor` e `carregar_ou_inicializar_dataset`),
         nunca de forma silenciosa.
    """
    if features_audio.ndim != 1 or vetor_texto.ndim != 1:
        raise ExtracaoFeaturesError("Vetores de áudio e de texto devem ser 1-D antes da fusão.")
    vetor = np.concatenate([features_audio, vetor_texto]).astype(np.float64)
    return np.nan_to_num(vetor, nan=0.0, posinf=0.0, neginf=0.0)


def validar_dimensao_vetor(vetor_multimodal: np.ndarray, dim_texto: int) -> None:
    esperado = len(nomes_colunas_features(dim_texto))
    if len(vetor_multimodal) != esperado:
        raise ExtracaoFeaturesError(
            f"Dimensão do vetor multimodal ({len(vetor_multimodal)}) não corresponde "
            f"ao número de colunas de features esperado ({esperado})."
        )
