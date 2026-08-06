"""
modelagem_base.py — Datathon Passos Mágicos (FIAP PósTech DTAT — Fase 5)
=======================================================================
A partir da base tidy (uma linha por aluno-ano), constrói a BASE DE
MODELAGEM: uma linha por transição (aluno, ano N -> ano N+1), unindo as
FEATURES do ano N com os DESFECHOS do ano N+1 pelo `ra`.

Isso operacionaliza o modelo como SISTEMA DE ALERTA PRECOCE: prever, a
partir do retrato do aluno hoje, seu risco no próximo ciclo.

Três rótulos são derivados (a escolha final do alvo fica no notebook):
  - y_risco           : defasagem_next < 0        (estará defasado em N+1) [alvo A]
  - y_piora_defasagem : defasagem_next < defas_N  (defasagem piora)
  - y_queda_inde      : inde_next < inde_N         (desempenho global cai)

Cortes oficiais de PEDRA (dicionário de dados) usados p/ validar INDE:
  Quartzo 2.405–5.506 · Ágata 5.506–6.868 · Ametista 6.868–8.230 · Topázio 8.230–9.294
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIDY_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "base_tidy.csv")
OUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "base_modelagem.csv")

FEATURES_N = [
    "fase_num", "idade", "genero", "ano_ingresso", "inde", "pedra", "defasagem",
    "iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian", "mat", "por", "ing",
]
DESFECHOS = ["defasagem", "inde", "ian", "pedra"]

PEDRA_CORTES = [  # (limite_inf, limite_sup, nome)
    (2.405, 5.506, "Quartzo"),
    (5.506, 6.868, "Ágata"),
    (6.868, 8.230, "Ametista"),
    (8.230, 9.294, "Topázio"),
]


def pedra_por_inde(inde: float):
    if pd.isna(inde):
        return np.nan
    for lo, hi, nome in PEDRA_CORTES:
        if lo <= inde <= hi:
            return nome
    return "Topázio" if inde > 9.294 else "Quartzo"  # extrapola extremos


def constroi_base_modelagem(salvar: bool = True) -> pd.DataFrame:
    tidy = pd.read_csv(TIDY_PATH)
    pares = []
    for ano_n, ano_next in [(2022, 2023), (2023, 2024)]:
        base = tidy[tidy.ano == ano_n].copy()
        nxt = tidy[tidy.ano == ano_next][["ra"] + DESFECHOS].copy()
        nxt = nxt.rename(columns={c: f"{c}_next" for c in DESFECHOS})
        m = base.merge(nxt, on="ra", how="inner")  # só quem existe nos 2 anos
        m["ano_base"] = ano_n
        m["ano_alvo"] = ano_next
        pares.append(m)

    df = pd.concat(pares, ignore_index=True)

    # ----- rótulos derivados -----
    df["y_risco"] = (df["defasagem_next"] < 0).astype(int)              # alvo A
    df["y_piora_defasagem"] = (df["defasagem_next"] < df["defasagem"]).astype(int)
    df["y_queda_inde"] = (df["inde_next"] < df["inde"]).astype(int)

    # marca linhas sem features utilizáveis (aluno era 'INCLUIR' no ano N)
    df["features_ausentes"] = df[["inde", "ian", "ida"]].isna().all(axis=1)

    cols = (["ra", "ano_base", "ano_alvo", "nome"] + FEATURES_N
            + [f"{c}_next" for c in DESFECHOS]
            + ["y_risco", "y_piora_defasagem", "y_queda_inde", "features_ausentes"])
    df = df[cols].sort_values(["ra", "ano_base"]).reset_index(drop=True)

    if salvar:
        df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = constroi_base_modelagem()
    print(f"Base de modelagem: {df.shape[0]} pares (N->N+1) x {df.shape[1]} colunas")
    print(f"  2022->2023: {(df.ano_base==2022).sum()}  |  2023->2024: {(df.ano_base==2023).sum()}")
    print(f"  Pares com features ausentes (aluno novo em N): {df.features_ausentes.sum()}")
    print(f"\nBalanceamento dos alvos (entre pares COM features):")
    ok = df[~df.features_ausentes]
    for y in ["y_risco", "y_piora_defasagem", "y_queda_inde"]:
        p = ok[y].mean()
        print(f"  {y:20s}: positivos={ok[y].sum():4d}/{len(ok)}  ({p:.1%})")
    print(f"\nSalvo em: {os.path.normpath(OUT_PATH)}")
