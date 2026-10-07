"""Senha do modo administrador."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets

from .config import SENHA_ADMIN_ITERACOES, SENHA_ADMIN_PATH, logger
from .dataset import _gravar_atomico


# ==============================================================================
# 11b. SENHA DO MODO ADMINISTRADOR
# ==============================================================================

def _hash_senha(senha: str, sal: bytes, iteracoes: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", senha.encode("utf-8"), sal, iteracoes).hex()


def senha_admin_definida() -> bool:
    return SENHA_ADMIN_PATH.is_file()


def definir_senha_admin(senha: str) -> None:
    """Grava a senha cifrada (PBKDF2-SHA256 com sal aleatório). A senha em si
    nunca é gravada -- só dá para conferir, não para ler."""
    sal = secrets.token_bytes(16)
    dados = {
        "algoritmo": "pbkdf2_sha256",
        "iteracoes": SENHA_ADMIN_ITERACOES,
        "sal": sal.hex(),
        "hash": _hash_senha(senha, sal, SENHA_ADMIN_ITERACOES),
    }
    SENHA_ADMIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _gravar_atomico(SENHA_ADMIN_PATH, lambda caminho: caminho.write_text(json.dumps(dados, indent=2), encoding="utf-8"))


def conferir_senha_admin(senha: str) -> bool:
    try:
        dados = json.loads(SENHA_ADMIN_PATH.read_text(encoding="utf-8"))
        calculado = _hash_senha(senha, bytes.fromhex(dados["sal"]), int(dados["iteracoes"]))
        return hmac.compare_digest(calculado, str(dados["hash"]))
    except Exception:
        logger.exception("Falha ao ler o arquivo de senha do administrador")
        return False
