"""
app.py — Sistema de Alerta Precoce de Defasagem · Passos Mágicos
================================================================
Datathon FIAP PósTech DTAT — Fase 5.

Disponibiliza o modelo de risco de defasagem como ferramenta de uso:
  1. Predição individual — estimar o risco de um aluno pelo seu retrato atual.
  2. Priorização da turma — ranquear a coorte mais recente (2024) por risco
     previsto para o próximo ciclo (a lista de quem intervir primeiro).
  3. Sobre o modelo — transparência: desempenho, variáveis e limites.

O modelo é treinado no startup (@st.cache_resource) a partir de src/modelo.py —
sem artefato .pkl, padrão que garante um deploy estável no Streamlit Cloud.
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))
from modelo import (treina_modelo, prepara_dados, split_temporal, novo_modelo,
                    FEATURES, FEATURE_LABELS)

TIDY_PATH = os.path.join(BASE_DIR, "data", "processed", "base_tidy.csv")

st.set_page_config(page_title="Passos Mágicos · Alerta de Defasagem",
                   page_icon="🎓", layout="wide")

# ------------------------------------------------------------------ cache
@st.cache_resource
def carrega_modelo():
    """Treina o modelo em todos os pares disponíveis (uso em produção)."""
    df = prepara_dados()
    modelo = treina_modelo(df)                       # treina em todos os 1.365 pares
    # métricas do teste temporal (para a aba de transparência)
    from sklearn.metrics import roc_auc_score
    Xtr, ytr, Xte, yte = split_temporal(df)
    m_eval = novo_modelo().fit(Xtr, ytr)
    auc = roc_auc_score(yte, m_eval.predict_proba(Xte)[:, 1])
    return modelo, auc, df


@st.cache_data
def carrega_coorte_atual():
    """Alunos de 2024 — a coorte a ser pontuada para o próximo ciclo."""
    tidy = pd.read_csv(TIDY_PATH)
    d = tidy[tidy["ano"] == 2024].copy()
    d["genero_bin"] = (d["genero"].astype(str).str.upper().str[0] == "M").astype(int)
    return d


def monta_features(linha: dict) -> pd.DataFrame:
    return pd.DataFrame([{f: linha.get(f, np.nan) for f in FEATURES}])[FEATURES]


def faixa_risco(p: float) -> tuple[str, str]:
    if p >= 0.66:  return "Alto", "🔴"
    if p >= 0.40:  return "Médio", "🟡"
    return "Baixo", "🟢"


modelo, auc_temporal, df_pares = carrega_modelo()

# ------------------------------------------------------------------ header
st.title("🎓 Sistema de Alerta Precoce de Defasagem")
st.caption("Associação Passos Mágicos · modelo preditivo de risco para o próximo ciclo letivo")

with st.expander("Sobre o projeto", expanded=False):
    st.markdown(
        "A partir do retrato de um aluno em um ano, o modelo estima a probabilidade de ele entrar "
        "em **defasagem no ciclo seguinte** — permitindo à Passos Mágicos **priorizar a intervenção "
        "antes da queda**. Treinado sobre a base PEDE (2022–2024) com validação temporal "
        f"(ROC-AUC {auc_temporal:.3f}). Três frentes: predição individual, priorização da turma e "
        "transparência do modelo."
    )

st.info(
    "🔒 **Ferramenta de apoio à decisão.** Os escores orientam o acolhimento pedagógico e psicossocial "
    "— **não substituem** a avaliação da equipe e **não devem rotular** o aluno. Trabalha com dados "
    "sensíveis (RA e indicadores): uso restrito e responsável.",
    icon="ℹ️",
)

aba1, aba2, aba3 = st.tabs(["🔎 Predição individual",
                            "📋 Priorização da turma",
                            "ℹ️ Sobre o modelo"])

# =================================================================== ABA 1
with aba1:
    st.subheader("Estimar o risco de um aluno")
    st.write("Preencha o retrato atual do aluno para estimar a probabilidade de ele "
             "entrar em **defasagem no próximo ciclo**.")
    col1, col2, col3 = st.columns(3)
    with col1:
        idade = st.number_input("Idade", 5, 30, 13)
        fase_num = st.number_input("Fase (0 = Alfa)", 0, 9, 3)
        genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
    with col2:
        defasagem = st.number_input("Defasagem atual (0 = em dia; negativo = atrasado)", -6, 3, 0)
        ano_ingresso = st.number_input("Ano de ingresso", 2016, 2024, 2022)
        ida = st.slider("IDA — desempenho acadêmico", 0.0, 10.0, 6.5, 0.1)
    with col3:
        ieg = st.slider("IEG — engajamento", 0.0, 10.0, 7.0, 0.1)
        iaa = st.slider("IAA — autoavaliação", 0.0, 10.0, 7.0, 0.1)
        ips = st.slider("IPS — psicossocial", 0.0, 10.0, 6.5, 0.1)
        ipv = st.slider("IPV — ponto de virada", 0.0, 10.0, 6.5, 0.1)

    if st.button("Calcular risco", type="primary"):
        linha = dict(idade=idade, fase_num=fase_num,
                     genero_bin=1 if genero == "Masculino" else 0,
                     ano_ingresso=ano_ingresso, defasagem=defasagem,
                     ida=ida, ieg=ieg, iaa=iaa, ips=ips, ipv=ipv)
        p = float(modelo.predict_proba(monta_features(linha))[:, 1][0])
        faixa, emoji = faixa_risco(p)
        c1, c2 = st.columns([1, 2])
        c1.metric("Probabilidade de defasagem", f"{p:.0%}")
        c1.markdown(f"### {emoji} Risco **{faixa}**")
        c2.progress(p)
        if faixa == "Alto":
            c2.warning("Aluno com alto risco de defasagem no próximo ciclo — "
                       "candidato prioritário a acompanhamento pedagógico.")
        elif faixa == "Médio":
            c2.info("Risco intermediário — vale monitorar de perto os indicadores de engajamento.")
        else:
            c2.success("Baixo risco no momento — manter o acompanhamento regular.")
        st.caption("Estimativa estatística de apoio à decisão; não substitui a avaliação "
                   "pedagógica e psicossocial da equipe.")

# =================================================================== ABA 2
with aba2:
    st.subheader("Priorização da coorte de 2024 para o próximo ciclo")
    st.write("O modelo pontua cada aluno de 2024 pelo risco de defasagem no ciclo seguinte. "
             "Use a lista para **priorizar a intervenção** onde ela mais evita a queda.")
    coorte = carrega_coorte_atual()
    Xc = coorte.reindex(columns=FEATURES)
    coorte["risco"] = modelo.predict_proba(Xc)[:, 1]
    coorte["faixa"] = coorte["risco"].apply(lambda p: faixa_risco(p)[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Alunos avaliados", len(coorte))
    c2.metric("Risco alto", int((coorte["faixa"] == "Alto").sum()))
    c3.metric("Já em dia mas em risco",
              int(((coorte["defasagem"] >= 0) & (coorte["risco"] >= 0.5)).sum()),
              help="Alunos hoje adequados que o modelo sinaliza para queda — onde a intervenção precoce mais rende.")

    apenas_em_dia = st.checkbox("Mostrar apenas alunos hoje em dia (foco em prevenção)", value=False)
    vis = coorte[coorte["defasagem"] >= 0] if apenas_em_dia else coorte
    vis = vis.sort_values("risco", ascending=False)
    tabela = vis[["ra", "fase_num", "idade", "defasagem", "ida", "ieg", "ipv", "risco", "faixa"]].copy()
    tabela["risco"] = (tabela["risco"] * 100).round(0).astype(int)
    tabela = tabela.rename(columns={"ra": "Aluno", "fase_num": "Fase", "idade": "Idade",
                                    "defasagem": "Defasagem", "risco": "Risco (%)", "faixa": "Faixa"})
    st.dataframe(tabela, use_container_width=True, hide_index=True, height=430)
    cexp1, cexp2 = st.columns([1, 2])
    cexp1.download_button("⬇️  Exportar lista priorizada (CSV)",
                          tabela.to_csv(index=False).encode("utf-8"),
                          "priorizacao_risco_defasagem.csv", "text/csv",
                          type="primary", use_container_width=True)
    cexp2.caption("Lista com dados sensíveis de alunos — compartilhe apenas com a equipe pedagógica "
                  "responsável e use como ponto de partida para o acolhimento, não como classificação.")

# =================================================================== ABA 3
with aba3:
    st.subheader("Sobre o modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (teste temporal)", f"{auc_temporal:.3f}")
    c2.metric("Pares de treino (N→N+1)", len(df_pares))
    c3.metric("Alvo", "Defasagem em N+1")
    st.markdown("""
**Como funciona.** A partir do retrato do aluno em um ano, o modelo estima a probabilidade de ele
estar defasado no ano seguinte. Foi validado com **split temporal** (aprende em 2022→2023, é testado
em 2023→2024), simulando o uso real de prever uma coorte futura.

**O que pesa na previsão.** A posição estrutural do aluno (fase, defasagem atual, idade) é o que mais
importa; entre os indicadores comportamentais, o **ponto de virada (IPV)** lidera. O gênero é
irrelevante para o modelo — não há discriminação por sexo.

**Valor para a Passos Mágicos.** Mesmo entre alunos hoje em dia, o modelo separa bem quem tende a
cair: priorizar o grupo de maior risco previsto rende cerca de **3× a taxa-base** de acerto —
concentrando os recursos de intervenção onde eles mais evitam a defasagem.

**Limites.** É uma ferramenta de apoio à decisão, baseada em padrões históricos (2022–2024). Não
substitui o olhar pedagógico e psicossocial da equipe, e deve ser revisada a cada novo ciclo de dados.
""")
    with st.expander("Importância das variáveis"):
        from sklearn.inspection import permutation_importance
        Xtr, ytr, Xte, yte = split_temporal(df_pares)
        m_eval = novo_modelo().fit(Xtr, ytr)
        pi = permutation_importance(m_eval, Xte, yte, n_repeats=10, random_state=42, scoring="roc_auc")
        imp = (pd.Series(pi.importances_mean, index=[FEATURE_LABELS[f] for f in FEATURES])
               .sort_values(ascending=False))
        st.bar_chart(imp)
