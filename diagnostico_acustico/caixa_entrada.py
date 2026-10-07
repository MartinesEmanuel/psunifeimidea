"""Caixa de entrada: casos gravados nos celulares, à espera da decisão do administrador.

Cada caso recebido pela caixa postal fica numa pasta caixa_de_entrada/<id>/:
    caso.json      o que o celular enviou (técnico, modelo da máquina, laudo, resultado do app...)
    audio.wav      a gravação original
    estado.json    o que aconteceu aqui: análise da Central, decisão e envio da resposta

A IA só aprende com o caso depois que o administrador o aprova. O técnico
recebe duas respostas, cada uma na sincronização seguinte ao fato:
    1. a análise da Central (som + laudo), sem esperar o administrador;
    2. a decisão final (aprovado ou rejeitado), quando o administrador decidir.
Depois que a decisão é enviada, o áudio sai da caixa de entrada (se aprovado,
ele já está na base de conhecimento) e fica só o histórico.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import CAIXA_ENTRADA_DIR, FORMATO_CASO_CELULAR, Etapa, _sem_progresso, logger
from .dataset import _gravar_atomico, carregar_ou_inicializar_dataset, rotulos_do_dataset
from .erros import PipelineError
from .features import montar_vetor_multimodal
from .pipeline import ResultadoAbsorcao, absorver_caso, calcular_hash_video, processar_video_e_extrair
from .recursos import _trava_processamento, obter_dimensao_vetor_texto
from .texto import categorias_ignoradas_no_laudo, extrair_vetor_texto, limpar_texto, remover_termos_de_problema
from .treino import EstadoModelo, diagnosticar

_ID_VALIDO = re.compile(r"^[A-Za-z0-9-]{8,64}$")
TAMANHO_MAXIMO_CASO = 200 * 1024 * 1024  # 200 MB


class CasoInvalidoError(PipelineError):
    """O pacote enviado pelo celular não está no formato esperado."""


def _pasta(caso_id: str) -> Path:
    if not _ID_VALIDO.match(str(caso_id)):
        raise CasoInvalidoError(f"identificador de caso inválido: {caso_id!r}")
    return CAIXA_ENTRADA_DIR / str(caso_id)


def _ler_json(caminho: Path) -> Dict[str, Any]:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}


def _gravar_json(caminho: Path, dados: Dict[str, Any]) -> None:
    _gravar_atomico(caminho, lambda c: c.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8"))


def receber_caso(linha: Dict[str, Any], zip_bytes: bytes) -> bool:
    """Guarda um caso baixado da caixa postal. Devolve False se ele já estava
    aqui (a sincronização pode ser repetida sem duplicar nada)."""
    caso_id = str(linha.get("id", ""))
    pasta = _pasta(caso_id)
    if (pasta / "caso.json").exists():
        return False
    if len(zip_bytes) > TAMANHO_MAXIMO_CASO:
        raise CasoInvalidoError(f"o caso {caso_id} é grande demais ({len(zip_bytes) / 1e6:.0f} MB).")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            caso = json.loads(zf.read("caso.json").decode("utf-8"))
            nome_audio = str((caso.get("gravacao") or {}).get("arquivo") or "audio.wav")
            audio = zf.read(nome_audio)
    except (KeyError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CasoInvalidoError(f"o pacote do caso {caso_id} está incompleto ou corrompido ({exc}).") from exc
    if str(caso.get("id")) != caso_id:
        raise CasoInvalidoError(f"o pacote do caso {caso_id} traz outro identificador ({caso.get('id')}).")
    if int(caso.get("formato", 0)) > FORMATO_CASO_CELULAR:
        raise CasoInvalidoError(f"o caso {caso_id} veio de uma versão mais nova do app; atualize o site.")
    if not audio:
        raise CasoInvalidoError(f"o caso {caso_id} veio sem áudio.")

    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "audio.wav").write_bytes(audio)
    caso["tecnico_nome"] = linha.get("tecnico_nome") or (caso.get("tecnico") or {}).get("nome") or ""
    _gravar_json(pasta / "estado.json", {"recebido_em": datetime.now().isoformat(timespec="seconds")})
    _gravar_json(pasta / "caso.json", caso)  # por último: só existe depois que tudo foi gravado
    return True


def listar_casos() -> List[Dict[str, Any]]:
    """Todos os casos, mais antigos primeiro, com caso.json + estado.json juntos."""
    if not CAIXA_ENTRADA_DIR.exists():
        return []
    casos = []
    for pasta in CAIXA_ENTRADA_DIR.iterdir():
        if not (pasta / "caso.json").exists():
            continue
        try:
            caso = _ler_json(pasta / "caso.json")
            caso["estado"] = _ler_json(pasta / "estado.json")
            caso["tem_audio"] = (pasta / "audio.wav").exists()
            casos.append(caso)
        except Exception:
            logger.exception("Caso ilegível na caixa de entrada: %s", pasta.name)
    return sorted(casos, key=lambda c: str(c.get("criado_em", "")))


def casos_pendentes() -> List[Dict[str, Any]]:
    return [c for c in listar_casos() if not c["estado"].get("decisao")]


def audio_do_caso(caso_id: str) -> Optional[bytes]:
    arquivo = _pasta(caso_id) / "audio.wav"
    return arquivo.read_bytes() if arquivo.exists() else None


def analisar_caso(caso_id: str, estado_modelo: Optional[EstadoModelo], etapa: Etapa = _sem_progresso) -> Dict[str, Any]:
    """Análise completa da Central (som + laudo, com remoção de voz) -- a
    "segunda opinião" que volta para o técnico. Sem IA treinada ainda, calcula
    só as características do som (o suficiente para aprovar o caso)."""
    pasta = _pasta(caso_id)
    caso = _ler_json(pasta / "caso.json")
    audio = audio_do_caso(caso_id)
    if audio is None:
        raise PipelineError("o áudio deste caso não está mais na caixa de entrada.")
    laudo = limpar_texto(caso.get("laudo") or "")
    with _trava_processamento():
        resultado_audio = processar_video_e_extrair(audio, "audio.wav", etapa)
        probs, termos_retirados = None, []
        if estado_modelo is not None:
            df = carregar_ou_inicializar_dataset(obter_dimensao_vetor_texto())
            termos = categorias_ignoradas_no_laudo([*rotulos_do_dataset(df), *estado_modelo.encoder.classes_])
            _, termos_retirados = remover_termos_de_problema(laudo, termos)
            vetor = montar_vetor_multimodal(resultado_audio.features_audio, extrair_vetor_texto(laudo, termos))
            probs = diagnosticar(vetor, estado_modelo.modelo, estado_modelo.encoder)
    (pasta / "audio_processado.wav").write_bytes(resultado_audio.wav_bytes)
    estado = _ler_json(pasta / "estado.json")
    estado["analise_central"] = {
        "probabilidades": probs,
        "termos_retirados": sorted({t.casefold() for t in termos_retirados}),
        "avisos": list(resultado_audio.avisos),
        "features_audio": resultado_audio.features_audio.tolist(),
        "duracao_s": resultado_audio.metadados.duracao_s,
        "analisado_em": datetime.now().isoformat(timespec="seconds"),
    }
    _gravar_json(pasta / "estado.json", estado)
    return estado["analise_central"]


def _decidir(caso_id: str, decisao: Dict[str, Any]) -> None:
    pasta = _pasta(caso_id)
    estado = _ler_json(pasta / "estado.json")
    if estado.get("decisao"):
        raise PipelineError("este caso já foi decidido.")
    estado["decisao"] = {**decisao, "decidido_em": datetime.now().isoformat(timespec="seconds")}
    _gravar_json(pasta / "estado.json", estado)


def aprovar_caso(caso_id: str, categoria: str, peca: str = "", observacao: str = "") -> ResultadoAbsorcao:
    """Coloca o caso na base de conhecimento com a categoria escolhida e retreina a IA."""
    pasta = _pasta(caso_id)
    caso = _ler_json(pasta / "caso.json")
    estado = _ler_json(pasta / "estado.json")
    analise = estado.get("analise_central")
    if not analise or not (pasta / "audio_processado.wav").exists():
        raise PipelineError("analise o caso com a Central antes de aprovar.")
    if not limpar_texto(categoria):
        raise PipelineError("escolha a categoria do caso.")
    audio_original = audio_do_caso(caso_id) or b""
    absorcao = absorver_caso(
        categoria,
        np.array(analise["features_audio"], dtype=np.float64),
        caso.get("laudo") or "",
        (pasta / "audio_processado.wav").read_bytes(),
        hash_video=calcular_hash_video(audio_original),
        nome_arquivo=f"celular {caso_id}",
        modelo_maquina=caso.get("modelo_maquina") or "",
        origem="celular",
        peca=peca,
    )
    _decidir(caso_id, {
        "status": "aprovado", "categoria": absorcao.rotulo, "peca": limpar_texto(peca),
        "observacao": limpar_texto(observacao), "registro_id": absorcao.registro_id,
    })
    return absorcao


def rejeitar_caso(caso_id: str, observacao: str = "") -> None:
    """O caso não entra na base (gravação ruim, informação errada...)."""
    _decidir(caso_id, {"status": "rejeitado", "observacao": limpar_texto(observacao)})


def analises_pendentes() -> List[Dict[str, Any]]:
    """Casos ainda sem decisão, já analisados pela Central (com IA treinada),
    cuja análise ainda não foi para o técnico."""
    return [
        c for c in listar_casos()
        if not c["estado"].get("decisao")
        and (c["estado"].get("analise_central") or {}).get("probabilidades")
        and not c["estado"].get("analise_enviada_em")
    ]


def analise_para_o_celular(caso: Dict[str, Any]) -> Dict[str, Any]:
    """Primeira resposta ao técnico: a análise da Central, antes da decisão."""
    analise = caso["estado"]["analise_central"]
    return {
        "status": "em_analise",
        "categoria_final": None,
        "peca": None,
        "observacao": None,
        "analise_central": {"probabilidades": analise.get("probabilidades")},
        "analisado_em": analise.get("analisado_em"),
        "decidido_em": None,
    }


def marcar_analise_enviada(caso_id: str) -> None:
    pasta = _pasta(caso_id)
    estado = _ler_json(pasta / "estado.json")
    estado["analise_enviada_em"] = datetime.now().isoformat(timespec="seconds")
    _gravar_json(pasta / "estado.json", estado)


def respostas_pendentes() -> List[Dict[str, Any]]:
    """Casos decididos cuja decisão ainda não voltou para o técnico."""
    return [c for c in listar_casos() if c["estado"].get("decisao") and not c["estado"].get("resposta_enviada_em")]


def resposta_para_o_celular(caso: Dict[str, Any]) -> Dict[str, Any]:
    """Resposta final ao técnico: a decisão e a análise da Central."""
    decisao = caso["estado"]["decisao"]
    analise = caso["estado"].get("analise_central") or {}
    return {
        "status": decisao["status"],
        "categoria_final": decisao.get("categoria"),
        "peca": decisao.get("peca") or None,
        "observacao": decisao.get("observacao") or None,
        "analise_central": {"probabilidades": analise.get("probabilidades")} if analise else None,
        "analisado_em": analise.get("analisado_em"),
        "decidido_em": decisao.get("decidido_em"),
    }


def marcar_resposta_enviada(caso_id: str) -> None:
    """Registra o envio e tira os áudios da caixa de entrada (se o caso foi
    aprovado, o áudio já está guardado na base de conhecimento)."""
    pasta = _pasta(caso_id)
    estado = _ler_json(pasta / "estado.json")
    estado["resposta_enviada_em"] = datetime.now().isoformat(timespec="seconds")
    _gravar_json(pasta / "estado.json", estado)
    for nome in ("audio.wav", "audio_processado.wav"):
        (pasta / nome).unlink(missing_ok=True)
