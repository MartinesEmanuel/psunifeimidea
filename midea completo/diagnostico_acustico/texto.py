"""Laudo textual: limpeza, nomes de categorias ignorados e vetor do spaCy."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .erros import ExtracaoFeaturesError, PipelineError
from .recursos import carregar_spacy, obter_dimensao_vetor_texto


# ==============================================================================
# 6. PROCESSAMENTO DE TEXTO (vetor fixo via spaCy -- SEM TF-IDF)
# ==============================================================================

def limpar_texto(texto: str) -> str:
    """Normalização mínima e segura do laudo textual (sem alterar semântica)."""
    return " ".join((texto or "").strip().split())


_PALAVRA = re.compile(r"\w+")


def categorias_ignoradas_no_laudo(rotulos: Sequence[str]) -> List[str]:
    """LISTA DE CATEGORIAS IGNORADAS NA LEITURA DO LAUDO: o nome de cada
    categoria de problema cadastrada. Se o laudo traz o nome da categoria, a IA
    aprenderia a ler a resposta no texto em vez de ouvir o som. Como a lista
    vem das categorias da base, toda categoria nova entra nela automaticamente.
    Nomes mais longos vêm primeiro ("Folga no suporte" antes de "Folga")."""
    nomes = {limpar_texto(r) for r in rotulos if limpar_texto(r)}
    return sorted(nomes, key=lambda nome: (-len(nome), nome))


def _sem_acento_minusculo(texto: str) -> str:
    """Minúsculas e sem acento, com cada letra na MESMA posição do texto
    original -- assim o trecho encontrado pode ser recortado do original."""
    saida = []
    for c in texto:
        base = "".join(x for x in unicodedata.normalize("NFKD", c) if not unicodedata.combining(x)).lower()
        saida.append(base if len(base) == 1 else c)
    return "".join(saida)


def _padrao_palavra(palavra: str) -> str:
    """Aceita a palavra no singular ou no plural ("rolamento(s)", "vibração/vibrações")."""
    formas = {palavra, palavra + "s", palavra + "es"}
    if palavra.endswith("ao"):
        formas |= {palavra[:-2] + "oes", palavra[:-2] + "aes"}
    if palavra.endswith(("al", "el", "ol", "ul")):
        formas.add(palavra[:-1] + "is")
    return "(?:" + "|".join(re.escape(f) for f in sorted(formas, key=len, reverse=True)) + ")"


def remover_termos_de_problema(texto: str, categorias: Sequence[str]) -> Tuple[str, List[str]]:
    """Procura no laudo o nome de cada categoria da lista (inteiro, sem
    diferenciar maiúsculas, acentos e plural) e o retira. Devolve o texto que
    vai para a IA e os trechos retirados (para mostrar ao usuário)."""
    texto = texto or ""
    normalizado = _sem_acento_minusculo(texto)
    trechos: List[Tuple[int, int]] = []
    for nome in categorias:
        palavras = _PALAVRA.findall(_sem_acento_minusculo(nome))
        if not palavras:
            continue
        padrao = r"(?<!\w)" + r"\W+".join(_padrao_palavra(p) for p in palavras) + r"(?!\w)"
        trechos += [(m.start(), m.end()) for m in re.finditer(padrao, normalizado)]

    partes: List[str] = []
    retiradas: List[str] = []
    fim_anterior = 0
    for inicio, fim in sorted(trechos, key=lambda t: (t[0], -t[1])):  # mesmo início: o nome mais longo
        if inicio < fim_anterior:  # sobreposto a um nome mais longo já retirado
            continue
        partes.append(texto[fim_anterior:inicio])
        retiradas.append(texto[inicio:fim])
        fim_anterior = fim
    partes.append(texto[fim_anterior:])
    return limpar_texto(" ".join(partes)), retiradas


def extrair_vetor_texto(texto: str, termos_ignorados: Optional[Sequence[str]] = None) -> np.ndarray:
    """MULTIMODALIDADE FIXA: usa EXCLUSIVAMENTE `doc.vector` do spaCy
    (pt_core_news_sm). Diferente do TF-IDF -- cujo vocabulário (e, portanto, o
    número de colunas) cresce a cada palavra inédita, gerando shape mismatch em
    aprendizado incremental -- o `doc.vector` tem SEMPRE o mesmo tamanho fixo,
    não importa o texto de entrada.

    Se `termos_ignorados` for informado, os nomes das categorias cadastradas
    são retirados do texto antes (ver `remover_termos_de_problema`)."""
    try:
        nlp = carregar_spacy()
        dim = obter_dimensao_vetor_texto()
        texto_limpo = limpar_texto(texto)
        if termos_ignorados:
            texto_limpo, _ = remover_termos_de_problema(texto_limpo, termos_ignorados)
        if not texto_limpo:
            return np.zeros(dim, dtype=np.float64)
        doc = nlp(texto_limpo)
        return doc.vector.astype(np.float64)
    except PipelineError:
        raise
    except Exception as exc:
        raise ExtracaoFeaturesError(f"falha ao extrair o vetor textual: {exc}") from exc
