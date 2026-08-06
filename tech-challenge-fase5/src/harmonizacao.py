"""
harmonizacao.py — Datathon Passos Mágicos (FIAP PósTech DTAT — Fase 5)
=====================================================================
Lê as três abas do arquivo bruto PEDE (2022, 2023, 2024), que possuem
esquemas divergentes, e as unifica numa base TIDY: uma linha por
(aluno, ano). Este é o alicerce de toda a análise (EDA, storytelling,
modelo de risco e app Streamlit).

Decisões de harmonização documentadas:
  - `ra`        : chave do aluno (string "RA-N"), estável entre anos ->
                  permite ligar o mesmo aluno ao longo do tempo.
  - `fase_num`  : nível da fase extraído dos 3 formatos distintos
                  (2022 numérico; 2023 "FASE N"/ALFA; 2024 "1A".."9"/ALFA).
                  ALFA (alfabetização) -> 0.
  - `inde`      : coação para numérico; rótulo "INCLUIR" (aluno recém
                  incluído, ainda não avaliado) -> NaN.
  - `ipp`       : inexistente em 2022 (estrutural) -> NaN nesse ano.
  - indicadores : IAA, IEG, IPS, IPP, IDA, IPV, IAN mantidos como float.
  - `defasagem` : inteiro (fase atual - fase ideal); negativo = atrasado.
"""

from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "PEDE_PASSOS_DESAFIO.xlsx")
OUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "base_tidy.csv")

# Mapeamento por ano: {coluna_canonica: nome_no_arquivo}
COLMAP = {
    2022: {
        "ra": "RA", "fase_raw": "Fase", "turma": "Turma", "nome": "Nome",
        "idade": "Idade 22", "genero": "Gênero", "ano_ingresso": "Ano ingresso",
        "instituicao": "Instituição de ensino", "inde": "INDE 22", "pedra": "Pedra 22",
        "iaa": "IAA", "ieg": "IEG", "ips": "IPS", "ipp": None, "ida": "IDA",
        "ipv": "IPV", "ian": "IAN", "defasagem": "Defas", "fase_ideal": "Fase ideal",
        "mat": "Matem", "por": "Portug", "ing": "Inglês", "atingiu_pv": "Atingiu PV",
    },
    2023: {
        "ra": "RA", "fase_raw": "Fase", "turma": "Turma", "nome": "Nome Anonimizado",
        "idade": "Idade", "genero": "Gênero", "ano_ingresso": "Ano ingresso",
        "instituicao": "Instituição de ensino", "inde": "INDE 2023", "pedra": "Pedra 2023",
        "iaa": "IAA", "ieg": "IEG", "ips": "IPS", "ipp": "IPP", "ida": "IDA",
        "ipv": "IPV", "ian": "IAN", "defasagem": "Defasagem", "fase_ideal": "Fase Ideal",
        "mat": "Mat", "por": "Por", "ing": "Ing", "atingiu_pv": "Atingiu PV",
    },
    2024: {
        "ra": "RA", "fase_raw": "Fase", "turma": "Turma", "nome": "Nome Anonimizado",
        "idade": "Idade", "genero": "Gênero", "ano_ingresso": "Ano ingresso",
        "instituicao": "Instituição de ensino", "inde": "INDE 2024", "pedra": "Pedra 2024",
        "iaa": "IAA", "ieg": "IEG", "ips": "IPS", "ipp": "IPP", "ida": "IDA",
        "ipv": "IPV", "ian": "IAN", "defasagem": "Defasagem", "fase_ideal": "Fase Ideal",
        "mat": "Mat", "por": "Por", "ing": "Ing", "atingiu_pv": "Atingiu PV",
    },
}

INDICADORES = ["iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian"]
CANON_ORDER = (
    ["ra", "ano", "nome", "genero", "idade", "ano_ingresso", "instituicao",
     "fase_raw", "fase_num", "fase_ideal", "defasagem", "turma", "pedra", "inde"]
    + INDICADORES + ["mat", "por", "ing", "atingiu_pv"]
)


def _extrai_fase_num(v) -> float:
    """Extrai o nível numérico da fase dos 3 formatos. ALFA -> 0."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if "ALFA" in s:
        return 0.0
    m = re.search(r"\d+", s)          # pega o primeiro número (ex.: "1A"->1, "FASE 3"->3)
    return float(m.group()) if m else np.nan


def _limpa_pedra(v):
    """Normaliza rótulos de pedra; 'INCLUIR' e nulos viram NaN."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s.upper() in {"INCLUIR", "NAN", ""}:
        return np.nan
    # padroniza acentuação de Ágata
    return {"Agata": "Ágata", "AGATA": "Ágata"}.get(s, s)


def carrega_ano(ano: int) -> pd.DataFrame:
    """Lê uma aba e devolve o DataFrame já no esquema canônico."""
    m = COLMAP[ano]
    raw = pd.read_excel(RAW_PATH, sheet_name=f"PEDE{ano}")
    out = pd.DataFrame()
    for canon, origem in m.items():
        out[canon] = raw[origem] if (origem is not None and origem in raw.columns) else np.nan
    out["ano"] = ano

    # coações de tipo
    out["inde"] = pd.to_numeric(out["inde"], errors="coerce")
    out["idade"] = pd.to_numeric(out["idade"], errors="coerce")
    for c in INDICADORES + ["mat", "por", "ing", "defasagem"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["fase_num"] = out["fase_raw"].apply(_extrai_fase_num)
    out["pedra"] = out["pedra"].apply(_limpa_pedra)
    out["ra"] = out["ra"].astype(str).str.strip()
    # genero vem com rótulos divergentes entre anos -> normaliza p/ F/M
    _gmap = {"Menina": "F", "Feminino": "F", "Menino": "M", "Masculino": "M"}
    out["genero"] = out["genero"].map(lambda g: _gmap.get(str(g).strip(), np.nan))
    return out[CANON_ORDER]


def constroi_base_tidy(salvar: bool = True) -> pd.DataFrame:
    """Empilha os três anos numa base tidy (uma linha por aluno-ano)."""
    df = pd.concat([carrega_ano(a) for a in (2022, 2023, 2024)], ignore_index=True)
    df = df.sort_values(["ra", "ano"]).reset_index(drop=True)
    if salvar:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = constroi_base_tidy()
    print(f"Base tidy: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"Salvo em: {os.path.normpath(OUT_PATH)}")
