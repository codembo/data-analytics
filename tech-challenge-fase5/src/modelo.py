"""
modelo.py — Datathon Passos Mágicos (FIAP PósTech DTAT — Fase 5)
================================================================
Treino do modelo de RISCO DE DEFASAGEM (pergunta 9). Função única de
treino usada tanto pelo notebook (avaliação) quanto pelo app Streamlit
(treino no startup via @st.cache_resource — sem artefato .pkl, padrão
que resolveu o deploy na Fase 4).

Decisões de modelagem (validadas na EDA e na discussão com a turma):
  - Split TEMPORAL: treina em pares 2022→2023, testa em 2023→2024
    (simula uso real: aprender no passado, prever o próximo ciclo).
  - IAN é função determinística da defasagem -> excluído (redundante).
  - INDE/PEDRA e Mat/Por/Ing excluídos (redundantes com indicadores/IDA).
  - IPP excluído: estruturalmente ausente em 2022 (todo o bloco de treino).
  - HistGradientBoosting: lida com valores nulos nativamente.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOD_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "base_modelagem.csv")

FEATURES = ["idade", "fase_num", "genero_bin", "ano_ingresso",
            "defasagem", "ida", "ieg", "iaa", "ips", "ipv"]
TARGET = "y_risco"

# rótulos amigáveis p/ exibição no app
FEATURE_LABELS = {
    "idade": "Idade", "fase_num": "Fase", "genero_bin": "Gênero (M=1)",
    "ano_ingresso": "Ano de ingresso", "defasagem": "Defasagem atual",
    "ida": "IDA (desempenho)", "ieg": "IEG (engajamento)",
    "iaa": "IAA (autoavaliação)", "ips": "IPS (psicossocial)",
    "ipv": "IPV (ponto de virada)",
}


def prepara_dados(caminho: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(caminho or MOD_PATH)
    df = df[~df["features_ausentes"]].copy()
    df["genero_bin"] = (df["genero"].astype(str).str.upper().str[0] == "M").astype(int)
    return df


def split_temporal(df: pd.DataFrame):
    tr = df[df["ano_base"] == 2022]
    te = df[df["ano_base"] == 2023]
    return (tr[FEATURES], tr[TARGET], te[FEATURES], te[TARGET])


def novo_modelo() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=42, max_iter=300, learning_rate=0.05, max_depth=4
    )


def treina_modelo(df: pd.DataFrame | None = None, so_treino_temporal: bool = False):
    """Treina e devolve o modelo. Para o app, treina em TODOS os pares
    disponíveis (mais dados = melhor inferência do próximo ciclo)."""
    df = df if df is not None else prepara_dados()
    if so_treino_temporal:
        X, y, *_ = split_temporal(df)
    else:
        X, y = df[FEATURES], df[TARGET]
    return novo_modelo().fit(X, y)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    df = prepara_dados()
    Xtr, ytr, Xte, yte = split_temporal(df)
    m = novo_modelo().fit(Xtr, ytr)
    print(f"AUC teste temporal: {roc_auc_score(yte, m.predict_proba(Xte)[:,1]):.3f}")
