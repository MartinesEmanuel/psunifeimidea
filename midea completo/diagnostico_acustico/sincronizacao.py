"""Sincronização com a caixa postal na internet (Supabase).

O site e o app funcionam sem internet. A caixa postal só é usada no momento da
troca, e os dois lados não precisam estar ligados ao mesmo tempo:

    celular -> caixa postal -> site   casos gravados no campo (vão para a caixa de entrada)
    site -> caixa postal -> celular   a análise da Central (logo na primeira troca), a decisão
                                      do administrador (quando ele decidir) e os pacotes com a IA
    site <-> caixa postal <-> site    a base de casos, entre os computadores do site
                                      (veja base_compartilhada.py)

Os celulares falam com um computador só, a "Central"; os outros computadores
do site só trocam a base de casos.

A caixa postal guarda os arquivos só durante a entrega: assim que o site baixa
um caso, o arquivo é apagado de lá. A base oficial continua neste computador.
Como preparar a caixa postal: veja nuvem/COMO_CONFIGURAR.txt.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote, urlsplit

import requests

from . import base_compartilhada, caixa_entrada, pacote
from .caixa_entrada import CasoInvalidoError
from .config import BUCKET_CAIXA_POSTAL, SINCRONIZACAO_PATH, logger
from .dataset import _gravar_atomico
from .erros import PipelineError
from .treino import EstadoModelo, carregar_modelo_salvo

ESTADO_SINCRONIZACAO_PATH = SINCRONIZACAO_PATH.with_name("sincronizacao_estado.json")
TEMPO_LIMITE = (8, 180)  # (conectar, transferir) em segundos

Progresso = Callable[[float, str], None]


class SincronizacaoError(PipelineError):
    """Falha ao falar com a caixa postal (sem internet, chave errada...)."""


class ArquivoAusenteError(SincronizacaoError):
    """Um arquivo esperado não está na caixa postal (afeta só aquele caso)."""


_LINK_DO_PAINEL = re.compile(r"supabase\.com/dashboard/project/([a-z0-9]+)", re.IGNORECASE)


def normalizar_url(url: str) -> str:
    """Deixa só o endereço do projeto (https://xxxx.supabase.co). Aceita o endereço
    copiado com "/rest/v1" no fim (como aparece numa das telas do Supabase) e até o
    link do painel do projeto."""
    texto = (url or "").strip()
    if not texto:
        return ""
    painel = _LINK_DO_PAINEL.search(texto)
    if painel:
        return f"https://{painel.group(1).lower()}.supabase.co"
    if "://" not in texto:
        texto = "https://" + texto
    partes = urlsplit(texto)
    return f"{partes.scheme}://{partes.netloc}" if partes.netloc else texto.rstrip("/")


@dataclass
class ConfigSincronizacao:
    url: str = ""
    chave_publica: str = ""
    chave_servico: str = ""
    # Central = o computador que recebe os casos dos celulares e envia a IA para eles.
    # Só um computador deve ser a Central; os outros só trocam a base de casos.
    central: bool = True

    def __post_init__(self) -> None:
        self.url = normalizar_url(self.url)

    @property
    def configurada(self) -> bool:
        return bool(self.url and self.chave_servico)


def codigo_conexao(config: ConfigSincronizacao) -> str:
    """Código que o técnico cola no app: endereço da caixa postal + chave PÚBLICA
    (nunca a de serviço). Formato: "DA1." + JSON em base64 (url-safe, sem "=")."""
    dados = json.dumps({"url": config.url, "chave": config.chave_publica}, separators=(",", ":"))
    return "DA1." + base64.urlsafe_b64encode(dados.encode("utf-8")).decode("ascii").rstrip("=")


def carregar_config() -> ConfigSincronizacao:
    if not SINCRONIZACAO_PATH.exists():
        return ConfigSincronizacao()
    try:
        dados = json.loads(SINCRONIZACAO_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Falha ao ler a configuração da sincronização")
        return ConfigSincronizacao()
    return ConfigSincronizacao(
        url=str(dados.get("url", "")),
        chave_publica=str(dados.get("chave_publica", "")).strip(),
        chave_servico=str(dados.get("chave_servico", "")).strip(),
        central=bool(dados.get("central", True)),
    )


def salvar_config(config: ConfigSincronizacao) -> None:
    SINCRONIZACAO_PATH.parent.mkdir(parents=True, exist_ok=True)
    dados = {"url": normalizar_url(config.url), "chave_publica": config.chave_publica.strip(),
             "chave_servico": config.chave_servico.strip(), "central": bool(config.central)}
    _gravar_atomico(SINCRONIZACAO_PATH, lambda c: c.write_text(json.dumps(dados, indent=2), encoding="utf-8"))


def carregar_estado() -> Dict[str, Any]:
    if not ESTADO_SINCRONIZACAO_PATH.exists():
        return {}
    try:
        return dict(json.loads(ESTADO_SINCRONIZACAO_PATH.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _salvar_estado(estado: Dict[str, Any]) -> None:
    ESTADO_SINCRONIZACAO_PATH.parent.mkdir(parents=True, exist_ok=True)
    _gravar_atomico(
        ESTADO_SINCRONIZACAO_PATH,
        lambda c: c.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8"),
    )


def pendencias() -> Dict[str, int]:
    """O que está esperando a próxima troca com a caixa postal."""
    return {
        "analises": len(caixa_entrada.analises_pendentes()),
        "respostas": len(caixa_entrada.respostas_pendentes()),
        "publicacoes": len(pacote.publicacoes_pendentes()),
    }


def _agora() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class CaixaPostal:
    """Acesso à caixa postal com a chave de serviço (só no computador da Central)."""

    def __init__(self, config: ConfigSincronizacao):
        self.url = config.url
        self.chave = config.chave_servico
        self.sessao = requests.Session()

    def _cabecalhos(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        cabecalhos = {"apikey": self.chave}
        if self.chave.startswith("eyJ"):  # chaves antigas do Supabase são tokens JWT
            cabecalhos["Authorization"] = f"Bearer {self.chave}"
        cabecalhos.update(extra or {})
        return cabecalhos

    def _pedir(self, metodo: str, caminho: str, *, headers: Optional[Dict[str, str]] = None,
               esperado=(200, 201, 204), **kwargs: Any) -> requests.Response:
        try:
            resposta = self.sessao.request(metodo, self.url + caminho, headers=self._cabecalhos(headers),
                                           timeout=TEMPO_LIMITE, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise SincronizacaoError("sem conexão com a internet, ou a caixa postal não respondeu.") from exc
        except requests.RequestException as exc:
            raise SincronizacaoError(f"falha ao falar com a caixa postal: {exc}") from exc
        if resposta.status_code in (401, 403):
            raise SincronizacaoError("a caixa postal recusou a chave de serviço. Confira as chaves na configuração.")
        if resposta.status_code == 404 and caminho.startswith("/rest/"):
            try:
                codigo = str(resposta.json().get("code") or "")
            except ValueError:
                codigo = ""
            if codigo in ("PGRST205", "42P01"):  # a tabela não existe
                raise SincronizacaoError(
                    "a caixa postal ainda não foi preparada. Rode o arquivo nuvem/supabase_configuracao.sql "
                    "no Supabase (veja nuvem/COMO_CONFIGURAR.txt)."
                )
            raise SincronizacaoError(
                f"a caixa postal não reconheceu o endereço (erro 404{' ' + codigo if codigo else ''}). Confira o "
                "endereço do projeto: deve ser só https://xxxx.supabase.co, sem nada depois."
            )
        if metodo == "GET" and caminho.startswith("/storage/") and resposta.status_code in (400, 404):
            raise ArquivoAusenteError(f"o arquivo {caminho.rsplit('/', 1)[-1]} não está na caixa postal.")
        if resposta.status_code not in esperado:
            raise SincronizacaoError(f"a caixa postal respondeu com erro {resposta.status_code}: {resposta.text[:300]}")
        return resposta

    def _objeto(self, caminho: str) -> str:
        return f"/storage/v1/object/{BUCKET_CAIXA_POSTAL}/{quote(caminho)}"

    # ---- operações
    def testar(self) -> None:
        self._pedir("GET", "/rest/v1/publicacoes", params={"select": "versao", "limit": "1"})

    def casos_novos(self) -> List[Dict[str, Any]]:
        return list(self._pedir("GET", "/rest/v1/casos", params={
            "status": "eq.enviado", "select": "*", "order": "criado_em.asc",
        }).json())

    def baixar(self, caminho: str) -> bytes:
        return self._pedir("GET", self._objeto(caminho)).content

    def apagar(self, caminhos: List[str]) -> None:
        self._pedir("DELETE", f"/storage/v1/object/{BUCKET_CAIXA_POSTAL}", json={"prefixes": list(caminhos)})

    def atualizar_caso(self, caso_id: str, campos: Dict[str, Any]) -> None:
        self._pedir("PATCH", "/rest/v1/casos", params={"id": f"eq.{caso_id}"}, json=campos,
                    headers={"Prefer": "return=minimal"})

    def enviar_arquivo(self, caminho: str, dados: bytes, tipo: str = "application/zip") -> None:
        self._pedir("POST", self._objeto(caminho), data=dados, headers={"Content-Type": tipo, "x-upsert": "true"})

    def registrar_publicacao(self, publicacao: Dict[str, Any]) -> None:
        self._pedir("POST", "/rest/v1/publicacoes", params={"on_conflict": "versao"}, json=publicacao,
                    headers={"Prefer": "resolution=merge-duplicates,return=minimal"})

    def listar_tecnicos(self) -> List[Dict[str, Any]]:
        dados = self._pedir("GET", "/auth/v1/admin/users", params={"per_page": "1000"}).json()
        usuarios = dados.get("users", []) if isinstance(dados, dict) else list(dados)
        return sorted(usuarios, key=lambda u: str(u.get("email", "")))

    def criar_tecnico(self, email: str, senha: str, nome: str) -> None:
        self._pedir("POST", "/auth/v1/admin/users", json={
            "email": email.strip(), "password": senha, "email_confirm": True,
            "user_metadata": {"nome": nome.strip(), "papel": "tecnico"},
        })


def testar_conexao(config: ConfigSincronizacao) -> None:
    CaixaPostal(config).testar()


def _trocar_base(caixa: CaixaPostal, estado_modelo: Optional[EstadoModelo], resumo: Dict[str, Any],
                 progresso: Progresso) -> Optional[EstadoModelo]:
    """Base de casos compartilhada com os outros computadores do site. Se chegaram
    casos novos (ou saíram casos), treina a IA de novo e devolve o modelo novo."""
    try:
        base = base_compartilhada.sincronizar_base(caixa, progresso)
    except SincronizacaoError as exc:
        resumo["avisos"].append(f"a troca da base com os outros computadores ficou para depois: {exc}")
        return estado_modelo
    resumo["base"] = base
    resumo["avisos"].extend(base["avisos"])
    if not (base["recebidos"] or base["excluidos"]):
        return estado_modelo
    progresso(1.0, "Treinando a IA com a base atualizada")
    from .pipeline import retreinar_agora  # aqui dentro: o pipeline importa este módulo indiretamente
    try:
        treinou, _, motivo = retreinar_agora()
    except PipelineError as exc:
        resumo["avisos"].append(f"a base foi atualizada, mas a IA não pôde ser treinada agora: {exc}")
        return estado_modelo
    if not treinou and motivo:
        resumo["avisos"].append(f"a IA não foi treinada de novo: {motivo}")
    return carregar_modelo_salvo()


def _trocar_com_celulares(caixa: CaixaPostal, estado_modelo: Optional[EstadoModelo], resumo: Dict[str, Any],
                          progresso: Progresso) -> None:
    """Só na Central: casos dos celulares, análises, decisões e pacotes com a IA."""
    # 1. Casos novos dos celulares -> caixa de entrada
    linhas = caixa.casos_novos()
    for i, linha in enumerate(linhas):
        caso_id = str(linha.get("id", ""))
        progresso(0.05 + 0.6 * i / max(1, len(linhas)), f"Recebendo o caso {i + 1} de {len(linhas)}")
        try:
            novo = caixa_entrada.receber_caso(linha, caixa.baixar(str(linha["arquivo"])))
        except ArquivoAusenteError as exc:
            resumo["avisos"].append(f"caso {caso_id}: {exc} Ele será procurado de novo na próxima sincronização.")
            continue
        except CasoInvalidoError as exc:
            resumo["avisos"].append(str(exc))
            caixa.atualizar_caso(caso_id, {"status": "rejeitado", "respondido_em": _agora(), "resposta": {
                "status": "rejeitado", "observacao": f"O pacote chegou com problema: {exc}"}})
            caixa.apagar([str(linha["arquivo"])])
            continue
        caixa.atualizar_caso(caso_id, {"status": "recebido"})
        caixa.apagar([str(linha["arquivo"])])
        resumo["recebidos"] += int(novo)
        if novo:
            progresso(0.05 + 0.6 * (i + 0.5) / max(1, len(linhas)), f"Analisando o caso {i + 1} na Central")
            try:
                caixa_entrada.analisar_caso(caso_id, estado_modelo)
                resumo["analisados"] += 1
            except PipelineError as exc:
                resumo["avisos"].append(f"o caso {caso_id} foi recebido, mas a análise falhou: {exc}")

    # 2. Análise da Central -> técnicos, sem esperar a decisão do administrador
    #    (o status continua "recebido"; a resposta vai com status "em_analise")
    analises = caixa_entrada.analises_pendentes()
    for i, caso in enumerate(analises):
        progresso(0.67 + 0.08 * i / max(1, len(analises)), f"Enviando análise {i + 1} de {len(analises)}")
        caixa.atualizar_caso(str(caso["id"]), {"resposta": caixa_entrada.analise_para_o_celular(caso)})
        caixa_entrada.marcar_analise_enviada(str(caso["id"]))
        resumo["analises_enviadas"] += 1

    # 3. Decisões do administrador -> técnicos (resposta final)
    respostas = caixa_entrada.respostas_pendentes()
    for i, caso in enumerate(respostas):
        progresso(0.75 + 0.12 * i / max(1, len(respostas)), f"Enviando decisão {i + 1} de {len(respostas)}")
        caixa.atualizar_caso(str(caso["id"]), {
            "status": caso["estado"]["decisao"]["status"],
            "resposta": caixa_entrada.resposta_para_o_celular(caso),
            "respondido_em": _agora(),
        })
        caixa_entrada.marcar_resposta_enviada(str(caso["id"]))
        resumo["respostas_enviadas"] += 1

    # 4. Pacotes novos (IA, textos, peças) -> celulares
    for publicacao in pacote.publicacoes_pendentes():
        progresso(0.9, f"Enviando o pacote versão {publicacao['versao']} para os celulares")
        caminho = f"publicacoes/{publicacao['arquivo']}"
        caixa.enviar_arquivo(caminho, pacote.caminho_pacote(publicacao).read_bytes())
        caixa.registrar_publicacao({
            "versao": int(publicacao["versao"]), "arquivo": caminho, "sha256": publicacao["sha256"],
            "tamanho": int(publicacao["tamanho"]), "notas": publicacao.get("notas") or None,
            "criado_em": publicacao["criado_em"],
        })
        pacote.marcar_publicacao_enviada(int(publicacao["versao"]))
        resumo["publicacoes_enviadas"] += 1


def sincronizar(estado_modelo: Optional[EstadoModelo], progresso: Progresso = lambda _f, _t: None) -> Dict[str, Any]:
    """Uma troca completa com a caixa postal. Cada parte concluída fica salva:
    se a internet cair no meio, a próxima sincronização continua de onde parou."""
    config = carregar_config()
    if not config.configurada:
        raise SincronizacaoError("a sincronização ainda não foi configurada.")
    caixa = CaixaPostal(config)
    resumo: Dict[str, Any] = {"recebidos": 0, "analisados": 0, "analises_enviadas": 0, "respostas_enviadas": 0,
                              "publicacoes_enviadas": 0, "avisos": [], "central": config.central, "base": None}
    estado = carregar_estado()
    try:
        progresso(0.02, "Conectando à caixa postal")
        caixa.testar()
        estado_modelo = _trocar_base(caixa, estado_modelo, resumo, lambda f, t: progresso(0.03 + 0.22 * f, t))
        if config.central:
            _trocar_com_celulares(caixa, estado_modelo, resumo, lambda f, t: progresso(0.27 + 0.73 * f, t))
    except SincronizacaoError as exc:
        estado.update({"ultima_tentativa": _agora(), "ultimo_erro": str(exc), "ultimo_resumo": resumo})
        _salvar_estado(estado)
        raise
    progresso(1.0, "Sincronização concluída")
    estado.update({"ultima_tentativa": _agora(), "ultima_sincronizacao": _agora(), "ultimo_erro": None,
                   "ultimo_resumo": resumo})
    _salvar_estado(estado)
    return resumo
