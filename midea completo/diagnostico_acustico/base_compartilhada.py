"""Base compartilhada entre os computadores do site, pela mesma caixa postal.

Cada computador continua funcionando sozinho, sem internet. Na sincronização:
    * envia os casos que foram alimentados nele (planilha + laudo + áudio);
    * recebe os casos alimentados nos outros computadores;
    * quando um caso é excluído num computador, ele sai dos outros também;
    * os textos dos defeitos também vão e voltam (vale o mais recente).
Depois de receber casos novos, cada computador treina a própria IA com a base
completa -- assim todos ficam com o mesmo conhecimento, sem trocar backups.

Na caixa postal (só o site acessa, com a chave de serviço; os técnicos não veem):
    tabela base_casos     uma linha por caso (dados da planilha + laudo)
    tabela base_textos    textos dos defeitos, por categoria
    base/<código>.flac    o áudio de cada caso (FLAC: mesmo som, metade do tamanho)
"""
from __future__ import annotations

import io
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

from .config import AUDIOS_DIR, BASE_COMPARTILHADA_PATH, TEXTOS_DIR, logger
from .conhecimento import CAMPOS_TEXTO, carregar_textos, salvar_textos_categoria
from .dataset import (
    COLUNAS_META,
    COLUNAS_META_EXTRAS,
    _caminho_relativo,
    _chave_rotulo,
    _gravar_atomico,
    _resolver_caminho,
    carregar_ou_inicializar_dataset,
    gerar_uuid_curto,
    modelos_do_dataset,
    normalizar_modelo,
    normalizar_rotulo,
    remover_arquivos_caso,
    rotulos_do_dataset,
    salvar_dataset_master,
)
from .features import nomes_colunas_features
from .recursos import _trava_processamento, obter_dimensao_vetor_texto

TABELA_CASOS = "/rest/v1/base_casos"
TABELA_TEXTOS = "/rest/v1/base_textos"
PASTA_AUDIOS = "base"
_LOTE = 20  # casos por pedido ao baixar

Progresso = Callable[[float, str], None]


def assinatura_caso(hash_video: Any, criado_em: Any) -> str:
    """Identifica o mesmo caso em computadores diferentes (o código curto pode se repetir
    entre computadores; o arquivo enviado + a hora do cadastro, não)."""
    return f"{'' if pd.isna(hash_video) else hash_video}|{'' if pd.isna(criado_em) else criado_em}"


def _assinatura_da_linha(linha: pd.Series) -> str:
    return assinatura_caso(linha.get("hash_video", ""), linha.get("criado_em", ""))


def _mesma_assinatura(a: str, b: str) -> bool:
    return not a.strip("|") or not b.strip("|") or a == b


def _nome_computador() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _em_lotes(itens: Sequence[Any], tamanho: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(itens), tamanho):
        yield itens[i:i + tamanho]


def _filtro_em(valores: Iterable[str]) -> str:
    return "in.(" + ",".join(json.dumps(str(v)) for v in valores) + ")"


# ------------------------------------------------ exclusões feitas aqui

def _ler_pendentes() -> Dict[str, str]:
    """código do caso -> assinatura, das exclusões ainda não avisadas aos outros computadores."""
    if not BASE_COMPARTILHADA_PATH.exists():
        return {}
    try:
        return dict(json.loads(BASE_COMPARTILHADA_PATH.read_text(encoding="utf-8")).get("exclusoes_pendentes", {}))
    except Exception:
        logger.exception("Falha ao ler %s", BASE_COMPARTILHADA_PATH.name)
        return {}


def _gravar_pendentes(pendentes: Dict[str, str]) -> None:
    BASE_COMPARTILHADA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dados = {"exclusoes_pendentes": dict(sorted(pendentes.items()))}
    _gravar_atomico(BASE_COMPARTILHADA_PATH,
                    lambda c: c.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8"))


def registrar_exclusoes(removidos: pd.DataFrame) -> None:
    """Chamado quando casos são excluídos neste computador: na próxima
    sincronização eles saem dos outros computadores também."""
    if removidos.empty:
        return
    pendentes = _ler_pendentes()
    for _, linha in removidos.iterrows():
        pendentes[str(linha["UUID"])] = _assinatura_da_linha(linha)
    _gravar_pendentes(pendentes)


# ------------------------------------------------ conversões

def _linha_para_json(linha: pd.Series) -> Dict[str, Any]:
    dados: Dict[str, Any] = {}
    for coluna, valor in linha.items():
        if isinstance(valor, (bool, np.bool_)):
            dados[str(coluna)] = bool(valor)
        elif isinstance(valor, (int, float, np.integer, np.floating)):
            dados[str(coluna)] = None if pd.isna(valor) else float(valor)
        else:
            dados[str(coluna)] = "" if valor is None or (isinstance(valor, float) and pd.isna(valor)) else str(valor)
    return dados


def _wav_para_flac(caminho: Path) -> bytes:
    info = sf.info(str(caminho))
    sinal, taxa = sf.read(str(caminho), dtype="float64", always_2d=False)
    subtipo = "PCM_16" if info.subtype in ("PCM_16", "PCM_U8", "PCM_S8") else "PCM_24"
    buffer = io.BytesIO()
    sf.write(buffer, sinal, taxa, format="FLAC", subtype=subtipo)
    return buffer.getvalue()


def _flac_para_wav(dados: bytes) -> bytes:
    info = sf.info(io.BytesIO(dados))
    sinal, taxa = sf.read(io.BytesIO(dados), dtype="float64", always_2d=False)
    buffer = io.BytesIO()
    sf.write(buffer, sinal, taxa, format="WAV", subtype="PCM_24" if info.subtype == "PCM_24" else "PCM_16")
    return buffer.getvalue()


def _destino_livre(pasta: Path, nome_salvo: str, codigo: str, extensao: str) -> Path:
    """Arquivo novo dentro de [pasta] (só o nome é aproveitado: nada de caminhos vindos de fora)."""
    nome = Path(str(nome_salvo or "").replace("\\", "/")).name
    if not nome.lower().endswith(extensao):
        nome = f"{codigo}{extensao}"
    destino = pasta / nome
    if destino.exists():
        destino = pasta / f"{Path(nome).stem}_{codigo}{extensao}"
    return destino


# ------------------------------------------------ caixa postal

def _listar_remotos(caixa) -> Dict[str, Tuple[bool, str]]:
    """código -> (excluído?, assinatura) de todos os casos na caixa postal (em páginas)."""
    remotos: Dict[str, Tuple[bool, str]] = {}
    inicio = 0
    while True:
        pagina = caixa._pedir("GET", TABELA_CASOS, params={
            "select": "uuid,excluido,assinatura", "order": "uuid.asc", "offset": str(inicio), "limit": "1000",
        }).json()
        for linha in pagina:
            remotos[str(linha["uuid"])] = (bool(linha.get("excluido")), str(linha.get("assinatura") or ""))
        if not pagina:
            return remotos
        inicio += len(pagina)


def _enviar_caso(caixa, linha: pd.Series, computador: str) -> None:
    codigo = str(linha["UUID"])
    arquivo = None
    audio = _resolver_caminho(linha["audio_path"]) if str(linha.get("audio_path", "")).strip() else None
    if audio is not None and audio.is_file():
        arquivo = f"{PASTA_AUDIOS}/{codigo}.flac"
        caixa.enviar_arquivo(arquivo, _wav_para_flac(audio), tipo="audio/flac")
    laudo = ""
    texto = _resolver_caminho(linha["text_path"]) if str(linha.get("text_path", "")).strip() else None
    if texto is not None and texto.is_file():
        laudo = texto.read_text(encoding="utf-8")
    caixa._pedir("POST", TABELA_CASOS, params={"on_conflict": "uuid"},
                 headers={"Prefer": "resolution=ignore-duplicates,return=minimal"},
                 json={"uuid": codigo, "dados": _linha_para_json(linha), "laudo": laudo, "arquivo": arquivo,
                       "computador": computador, "assinatura": _assinatura_da_linha(linha)})


def _baixar_lote(caixa, codigos: Sequence[str]) -> List[Dict[str, Any]]:
    linhas = caixa._pedir("GET", TABELA_CASOS, params={"select": "*", "uuid": _filtro_em(codigos)}).json()
    for linha in linhas:
        linha["audio_wav"] = None
        if linha.get("arquivo"):
            try:
                linha["audio_wav"] = _flac_para_wav(caixa.baixar(str(linha["arquivo"])))
            except Exception as exc:  # sem o áudio, o caso ainda serve para treinar
                logger.warning("Áudio do caso %s não veio: %s", linha.get("uuid"), exc)
    return linhas


# ------------------------------------------------ troca completa

def sincronizar_base(caixa, progresso: Progresso = lambda _f, _t: None) -> Dict[str, Any]:
    resumo: Dict[str, Any] = {"enviados": 0, "recebidos": 0, "excluidos": 0, "duplicados": 0, "renomeados": 0,
                              "textos_enviados": 0, "textos_recebidos": 0, "avisos": []}
    computador = _nome_computador()
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    progresso(0.0, "Conferindo a base dos outros computadores")
    remotos = _listar_remotos(caixa)

    # 1. Exclusões feitas aqui -> caixa postal (só se for o mesmo caso, não um código repetido)
    pendentes = _ler_pendentes()
    avisar = [c for c, assinatura in pendentes.items()
              if c in remotos and not remotos[c][0] and _mesma_assinatura(assinatura, remotos[c][1])]
    for lote in _em_lotes(avisar, 50):
        caixa._pedir("PATCH", TABELA_CASOS, params={"uuid": _filtro_em(lote)},
                     json={"excluido": True, "excluido_em": agora}, headers={"Prefer": "return=minimal"})
        for codigo in lote:
            remotos[codigo] = (True, remotos[codigo][1])
    if pendentes:
        _gravar_pendentes({})

    dim = obter_dimensao_vetor_texto()
    colunas_features = nomes_colunas_features(dim)
    with _trava_processamento():
        df = carregar_ou_inicializar_dataset(dim)
        mudou = False
        # 2. Código repetido entre computadores (casos diferentes): o daqui ganha um código novo
        for i, linha in df.iterrows():
            remoto = remotos.get(str(linha["UUID"]))
            if remoto and not _mesma_assinatura(_assinatura_da_linha(linha), remoto[1]):
                novo = gerar_uuid_curto(df)
                while novo in remotos:
                    novo = gerar_uuid_curto(df)
                df.at[i, "UUID"] = novo
                resumo["renomeados"] += 1
                mudou = True
        # 3. Exclusões feitas nos outros computadores -> aqui
        excluir = df["UUID"].map(lambda c: bool(remotos.get(str(c), (False, ""))[0]))
        if excluir.any():
            for _, linha in df[excluir].iterrows():
                remover_arquivos_caso(linha["audio_path"], linha["text_path"])
            resumo["excluidos"] = int(excluir.sum())
            df = df[~excluir].reset_index(drop=True)
            mudou = True
        if mudou:
            salvar_dataset_master(df)
        enviar = df[~df["UUID"].isin(list(remotos))]
        locais = set(df["UUID"])

    # 4. Casos daqui que os outros ainda não têm -> caixa postal
    for n, (_, linha) in enumerate(enviar.iterrows()):
        progresso(0.05 + 0.45 * n / max(1, len(enviar)), f"Enviando caso {n + 1} de {len(enviar)} para os outros computadores")
        _enviar_caso(caixa, linha, computador)
        resumo["enviados"] += 1

    # 5. Casos dos outros computadores -> aqui
    faltando = sorted(c for c, (excluido, _) in remotos.items() if not excluido and c not in locais)
    for n, lote in enumerate(_em_lotes(faltando, _LOTE)):
        progresso(0.5 + 0.45 * n * _LOTE / max(1, len(faltando)),
                  f"Recebendo casos dos outros computadores ({n * _LOTE + 1} a {n * _LOTE + len(lote)} de {len(faltando)})")
        baixados = _baixar_lote(caixa, lote)
        with _trava_processamento():
            df = carregar_ou_inicializar_dataset(dim)
            colunas = list(df.columns)
            rotulos, modelos = rotulos_do_dataset(df), modelos_do_dataset(df)
            hashes = {h for h in df["hash_video"] if str(h).strip()}
            codigos = set(df["UUID"])
            novas = []
            for item in baixados:
                dados = dict(item.get("dados") or {})
                codigo = str(item["uuid"])
                hash_video = str(dados.get("hash_video") or "")
                if codigo in codigos:
                    continue
                if hash_video and hash_video in hashes:  # o mesmo vídeo já foi cadastrado aqui
                    resumo["duplicados"] += 1
                    continue
                faltam = [c for c in colunas_features if dados.get(c) is None]
                if faltam:
                    resumo["avisos"].append(f"o caso {codigo} veio de um computador com outra versão do site "
                                            f"(faltam {len(faltam)} colunas) e ficou de fora.")
                    continue
                rotulo = normalizar_rotulo(str(dados.get("rotulo") or ""), rotulos)
                if rotulo not in rotulos:
                    rotulos.append(rotulo)
                modelo = str(dados.get("modelo_maquina") or "").strip()
                if modelo:
                    modelo = normalizar_modelo(modelo, modelos)
                    if modelo not in modelos:
                        modelos.append(modelo)
                texto = _destino_livre(TEXTOS_DIR, str(dados.get("text_path") or ""), codigo, ".txt")
                texto.write_text(str(item.get("laudo") or ""), encoding="utf-8")
                audio_rel = ""
                if item.get("audio_wav"):
                    audio = _destino_livre(AUDIOS_DIR, str(dados.get("audio_path") or ""), codigo, ".wav")
                    audio.write_bytes(item["audio_wav"])
                    audio_rel = _caminho_relativo(audio)
                linha = {c: dados.get(c) for c in colunas}
                for c in COLUNAS_META + COLUNAS_META_EXTRAS:
                    linha[c] = "" if linha.get(c) is None else str(linha[c])
                linha.update({"UUID": codigo, "rotulo": rotulo, "modelo_maquina": modelo,
                              "audio_path": audio_rel, "text_path": _caminho_relativo(texto)})
                novas.append(linha)
                codigos.add(codigo)
                if hash_video:
                    hashes.add(hash_video)
            if novas:
                df = pd.concat([df, pd.DataFrame(novas, columns=colunas)], ignore_index=True)
                salvar_dataset_master(df)
                resumo["recebidos"] += len(novas)

    # 6. Textos dos defeitos: vale o mais recente
    progresso(0.96, "Trocando os textos dos defeitos")
    remotos_textos = {str(r["chave"]): r for r in caixa._pedir(
        "GET", TABELA_TEXTOS, params={"select": "chave,categoria,textos,atualizado_em"}).json()}
    locais_textos = {_chave_rotulo(nome): (nome, campos) for nome, campos in carregar_textos().items()}
    for chave in sorted(set(remotos_textos) | set(locais_textos)):
        remoto, local = remotos_textos.get(chave), locais_textos.get(chave)
        data_remota = str(remoto.get("atualizado_em") or "") if remoto else ""
        data_local = str(local[1].get("atualizado_em") or "") if local else ""
        if local and data_local > data_remota:
            caixa._pedir("POST", TABELA_TEXTOS, params={"on_conflict": "chave"},
                         headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
                         json={"chave": chave, "categoria": local[0], "atualizado_em": data_local, "computador": computador,
                               "textos": {c: str(local[1].get(c, "")) for c in CAMPOS_TEXTO}})
            resumo["textos_enviados"] += 1
        elif remoto and data_remota > data_local:
            salvar_textos_categoria(str(remoto["categoria"]), dict(remoto.get("textos") or {}), atualizado_em=data_remota)
            resumo["textos_recebidos"] += 1
    progresso(1.0, "Base compartilhada em dia")
    return resumo
