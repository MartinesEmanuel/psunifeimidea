"""Extração e leitura de áudio; detecção e remoção de voz humana."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from .config import (
    CROSSFADE_S,
    DEMUCS_COBERTURA_AUDIO_TODO,
    DEMUCS_MARGEM_S,
    DEMUCS_UNIR_TRECHOS_S,
    Etapa,
    FFMPEG_TARGET_SR,
    FFMPEG_TIMEOUT_S,
    MIN_AUDIO_DURATION_S,
    MIN_SPEECH_RATIO,
    PICO_AUDIO_MUDO,
    RMS_AUDIO_BAIXO,
    VAD_SAMPLE_RATE,
    _sem_progresso,
    logger,
)
from .erros import ExtracaoAudioError
from .recursos import _silero_get_speech_timestamps, _threads_torch, carregar_demucs, carregar_vad


# ==============================================================================
# 4. E/S DE ÁUDIO (extração de vídeo, metadados, leitura de amostras)
# ==============================================================================

@dataclass(frozen=True)
class AudioMetadata:
    duracao_s: float
    sample_rate: int
    canais: int


def localizar_ffmpeg() -> Optional[str]:
    """Procura o ffmpeg no PATH e, no Windows, também nas pastas do winget --
    útil quando o ffmpeg acabou de ser instalado e o PATH desta janela ainda
    não foi atualizado. Se não houver ffmpeg no sistema, usa o que vem dentro
    do pacote pip `imageio-ffmpeg` (ver requirements.txt), o que permite rodar
    o app em qualquer computador sem instalar o ffmpeg à parte."""
    caminho = shutil.which("ffmpeg")
    if caminho:
        return caminho
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidatos = [local / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"]
        candidatos += sorted((local / "Microsoft" / "WinGet" / "Packages").glob("Gyan.FFmpeg*/*/bin/ffmpeg.exe"))
        for candidato in candidatos:
            if candidato.is_file():
                return str(candidato)
    try:
        import imageio_ffmpeg

        embutido = imageio_ffmpeg.get_ffmpeg_exe()
        if embutido and Path(embutido).is_file():
            return embutido
    except Exception:  # pacote ausente ou binário indisponível para esta plataforma
        pass
    return None


def _rodar_subprocesso(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, **kwargs,
    )


def _traduzir_erro_ffmpeg(stderr: Optional[str]) -> str:
    """Converte as mensagens mais comuns do ffmpeg em algo que o usuário entenda."""
    texto = (stderr or "").strip()
    baixo = texto.lower()
    if "matches no streams" in baixo or "does not contain any stream" in baixo:
        return "o arquivo não tem trilha de áudio (foi gravado sem som). Grave novamente com o microfone ativo."
    if "invalid data found" in baixo or "moov atom not found" in baixo:
        return "o arquivo parece corrompido, incompleto ou está num formato que o ffmpeg não reconhece."
    return f"falha ao extrair o áudio com o ffmpeg: {texto[-400:] or 'erro desconhecido'}"


def extrair_audio_de_video(video_bytes: bytes, sufixo_video: str) -> Path:
    """Salva o vídeo em um arquivo temporário e extrai a primeira trilha de
    áudio como .wav mono 44,1 kHz via ffmpeg. Quem chama deve apagar o .wav."""
    ffmpeg = localizar_ffmpeg()
    if ffmpeg is None:
        raise ExtracaoAudioError(
            "ffmpeg não encontrado. Rode o iniciar.bat de novo ou instale com: "
            "winget install ffmpeg -e --id Gyan.FFmpeg"
        )
    if not video_bytes:
        raise ExtracaoAudioError("o arquivo enviado está vazio (0 bytes).")

    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=sufixo_video, prefix="diag_video_")
    tmp_wav_path = Path(tempfile.gettempdir()) / f"diag_audio_{uuid.uuid4().hex}.wav"
    try:
        tmp_video.write(video_bytes)
        tmp_video.close()
        cmd = [
            ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-i", tmp_video.name,
            "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(FFMPEG_TARGET_SR), "-c:a", "pcm_s16le",
            str(tmp_wav_path),
        ]
        try:
            resultado = _rodar_subprocesso(cmd, FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            tmp_wav_path.unlink(missing_ok=True)
            raise ExtracaoAudioError("tempo limite excedido ao extrair o áudio com o ffmpeg.") from exc

        if resultado.returncode != 0 or not tmp_wav_path.exists() or tmp_wav_path.stat().st_size == 0:
            tmp_wav_path.unlink(missing_ok=True)
            raise ExtracaoAudioError(_traduzir_erro_ffmpeg(resultado.stderr))
        return tmp_wav_path
    finally:
        tmp_video.close()
        Path(tmp_video.name).unlink(missing_ok=True)


def obter_metadados_audio(caminho_wav: Path) -> AudioMetadata:
    """OTIMIZAÇÃO 1: usa EXCLUSIVAMENTE soundfile.info() (carregamento instantâneo,
    sem decodificar nenhuma amostra) para descobrir duração, sample_rate e canais."""
    try:
        info = sf.info(str(caminho_wav))
    except Exception as exc:
        raise ExtracaoAudioError(f"não foi possível ler metadados de '{caminho_wav.name}': {exc}") from exc
    duracao = info.frames / info.samplerate if info.samplerate else 0.0
    return AudioMetadata(duracao_s=duracao, sample_rate=info.samplerate, canais=info.channels)


def carregar_amostras_audio(caminho_wav: Path, sr_alvo: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Lê as amostras de áudio via soundfile (nunca librosa.load), converte para
    mono float32 e, se solicitado, reamostra para `sr_alvo` usando librosa.resample."""
    try:
        dados, sr = sf.read(str(caminho_wav), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ExtracaoAudioError(f"não foi possível ler as amostras de '{caminho_wav.name}': {exc}") from exc

    if dados.ndim > 1:
        dados = dados.mean(axis=1).astype(np.float32)

    if sr_alvo is not None and sr != sr_alvo:
        dados = librosa.resample(dados, orig_sr=sr, target_sr=sr_alvo).astype(np.float32)
        sr = sr_alvo

    return dados, sr


def validar_sinal_audio(y: np.ndarray, sr: int) -> List[str]:
    """Barra trilhas mudas ou curtas demais (que só gerariam features lixo) e
    devolve avisos não bloqueantes, como volume muito baixo."""
    duracao = len(y) / sr if sr else 0.0
    if duracao < MIN_AUDIO_DURATION_S:
        raise ExtracaoAudioError(
            f"o áudio tem só {duracao:.2f} s -- é preciso pelo menos {MIN_AUDIO_DURATION_S:.1f} s."
        )
    pico = float(np.max(np.abs(y))) if len(y) else 0.0
    if pico < PICO_AUDIO_MUDO:
        raise ExtracaoAudioError("a trilha de áudio está muda (silêncio absoluto). Verifique o microfone.")
    avisos: List[str] = []
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    if rms < RMS_AUDIO_BAIXO:
        avisos.append("volume muito baixo -- se possível, grave mais perto do equipamento.")
    return avisos


# ==============================================================================
# 4b. PRÉ-PROCESSAMENTO: DETECÇÃO (Silero VAD) E REMOÇÃO (Demucs) DE VOZ HUMANA
# ==============================================================================

def detectar_fala(y: np.ndarray, sr: int) -> Tuple[float, List[Tuple[float, float]]]:
    """OTIMIZAÇÃO 3: roda o Silero VAD apenas sobre o único áudio recém-extraído
    (nunca em loop varrendo pastas). Devolve a fração do áudio com fala e os
    trechos de fala, em segundos."""
    modelo_vad = carregar_vad()
    y16 = y if sr == VAD_SAMPLE_RATE else librosa.resample(y, orig_sr=sr, target_sr=VAD_SAMPLE_RATE)
    tensor_audio = torch.from_numpy(np.ascontiguousarray(y16, dtype=np.float32))
    with _threads_torch(1), torch.no_grad():
        timestamps = _silero_get_speech_timestamps(
            tensor_audio, modelo_vad, sampling_rate=VAD_SAMPLE_RATE, return_seconds=True
        )
    duracao = len(y16) / VAD_SAMPLE_RATE
    trechos = [(float(t["start"]), float(t["end"])) for t in timestamps]
    duracao_fala = sum(fim - inicio for inicio, fim in trechos)
    return (duracao_fala / duracao if duracao > 0 else 0.0), trechos


def _segmentos_para_demucs(trechos_fala: List[Tuple[float, float]], duracao: float) -> List[Tuple[float, float]]:
    """Agrupa os trechos de fala (com margem de contexto) nos intervalos que o
    Demucs deve processar. O resto do áudio fica intocado: além de mais rápido,
    evita artefatos de separação onde não havia voz."""
    segmentos: List[List[float]] = []
    for inicio, fim in sorted(trechos_fala):
        inicio, fim = max(0.0, inicio - DEMUCS_MARGEM_S), min(duracao, fim + DEMUCS_MARGEM_S)
        if segmentos and inicio - segmentos[-1][1] <= DEMUCS_UNIR_TRECHOS_S:
            segmentos[-1][1] = max(segmentos[-1][1], fim)
        else:
            segmentos.append([inicio, fim])
    cobertura = sum(fim - inicio for inicio, fim in segmentos)
    if duracao > 0 and cobertura / duracao >= DEMUCS_COBERTURA_AUDIO_TODO:
        return [(0.0, duracao)]
    return [(inicio, fim) for inicio, fim in segmentos]


def _separar_sem_voz(y: np.ndarray, sr: int, modelo: Any, apply_model: Callable[..., Any]) -> np.ndarray:
    """Equivalente a `demucs --two-stems vocals`: devolve a soma de todas as
    fontes exceto "vocals", em mono e no mesmo sample rate/tamanho da entrada."""
    sr_modelo = int(modelo.samplerate)
    x = y if sr == sr_modelo else librosa.resample(y, orig_sr=sr, target_sr=sr_modelo)
    wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    wav = wav.unsqueeze(0).expand(int(modelo.audio_channels), -1).contiguous()

    referencia = wav.mean(0)
    media, desvio = referencia.mean(), referencia.std()
    if float(desvio) < 1e-8:  # trecho silencioso: nada a separar
        return y.copy()

    with _threads_torch(os.cpu_count() or 1), torch.no_grad():
        fontes = apply_model(
            modelo, ((wav - media) / desvio)[None], device="cpu",
            shifts=0, split=True, overlap=0.25, progress=False,
        )[0]
    fontes = fontes * desvio + media

    indice_voz = list(modelo.sources).index("vocals")
    sem_voz = sum(fontes[i] for i in range(fontes.shape[0]) if i != indice_voz)
    sem_voz = sem_voz.mean(0).cpu().numpy().astype(np.float32)
    if sr != sr_modelo:
        sem_voz = librosa.resample(sem_voz, orig_sr=sr_modelo, target_sr=sr).astype(np.float32)

    if len(sem_voz) < len(y):
        sem_voz = np.pad(sem_voz, (0, len(y) - len(sem_voz)))
    return sem_voz[: len(y)]


def remover_voz_com_demucs(y: np.ndarray, sr: int, trechos_fala: List[Tuple[float, float]]) -> np.ndarray:
    """OTIMIZAÇÃO 3: separação de fontes pontual (Demucs) rodando só sobre este
    áudio -- e, dentro dele, só sobre os trechos em que o VAD encontrou fala."""
    modelo, apply_model = carregar_demucs()
    duracao = len(y) / sr
    saida = y.copy()
    n_fade = int(CROSSFADE_S * sr)

    for inicio_s, fim_s in _segmentos_para_demucs(trechos_fala, duracao):
        a, b = int(inicio_s * sr), min(len(y), int(round(fim_s * sr)))
        if b - a < int(0.1 * sr):
            continue
        trecho_sem_voz = _separar_sem_voz(y[a:b], sr, modelo, apply_model)
        f = min(n_fade, (b - a) // 2)
        if f > 0:  # transição suave com o áudio original nas bordas internas
            rampa = np.linspace(0.0, 1.0, f, dtype=np.float32)
            if a > 0:
                trecho_sem_voz[:f] = y[a:a + f] * (1 - rampa) + trecho_sem_voz[:f] * rampa
            if b < len(y):
                trecho_sem_voz[-f:] = trecho_sem_voz[-f:] * (1 - rampa) + y[b - f:b] * rampa
        saida[a:b] = trecho_sem_voz
    return saida


@dataclass
class ResultadoPreprocessamento:
    audio: np.ndarray
    ratio_fala: Optional[float]  # None = VAD indisponível
    demucs_aplicado: bool
    avisos: List[str] = field(default_factory=list)


def preprocessar_audio_pipeline(y: np.ndarray, sr: int, etapa: Etapa = _sem_progresso) -> ResultadoPreprocessamento:
    """Aplica VAD e, se necessário, Demucs -- SEMPRE de forma pontual sobre um
    único áudio. Se algum dos dois modelos estiver indisponível, o áudio
    original é usado e o motivo vai para a lista de avisos (degradação suave)."""
    etapa("detectando voz humana (Silero VAD)", 0.15)
    try:
        ratio_fala, trechos = detectar_fala(y, sr)
    except Exception as exc:
        logger.exception("Silero VAD falhou")
        return ResultadoPreprocessamento(
            audio=y, ratio_fala=None, demucs_aplicado=False,
            avisos=[f"a detecção de voz falhou ({exc}); o áudio foi usado sem remoção de voz."],
        )

    if ratio_fala < MIN_SPEECH_RATIO:
        logger.info("Sem voz relevante detectada (%.1f%%) -- Demucs pulado.", ratio_fala * 100)
        return ResultadoPreprocessamento(audio=y, ratio_fala=ratio_fala, demucs_aplicado=False)

    logger.info("Voz detectada (%.1f%% do áudio) -- aplicando Demucs.", ratio_fala * 100)
    etapa(f"removendo a voz humana detectada em {ratio_fala:.0%} do áudio (Demucs)", 0.3)
    try:
        audio_sem_voz = remover_voz_com_demucs(y, sr, trechos)
    except Exception as exc:
        logger.exception("Demucs falhou")
        return ResultadoPreprocessamento(
            audio=y, ratio_fala=ratio_fala, demucs_aplicado=False,
            avisos=[f"havia voz em {ratio_fala:.0%} do áudio, mas o Demucs falhou ({exc}); o áudio original foi usado."],
        )
    return ResultadoPreprocessamento(audio=audio_sem_voz, ratio_fala=ratio_fala, demucs_aplicado=True)
