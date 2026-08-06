# Datathon Passos Mágicos — Predição de Risco de Defasagem

Projeto da **Fase 5 (Tech Challenge / Datathon)** da PósTech FIAP em Data Analytics.
Caso: **Associação Passos Mágicos** — análise longitudinal do desenvolvimento educacional
(2022–2024) e modelo preditivo de risco de defasagem como sistema de alerta precoce.

## Como funciona

A base **PEDE** (Pesquisa Extensiva do Desenvolvimento Educacional) traz um retrato dos alunos
por ano. Usamos o `RA` para ligar o mesmo aluno entre anos e construir um problema **longitudinal**:
a partir do retrato do aluno no ano N, prever se ele estará **defasado no ano N+1** — permitindo à
ONG priorizar intervenção *antes* da queda.

## Estrutura

```
.
├── app.py                       # aplicação Streamlit (escore de risco por aluno)
├── requirements.txt
├── data/
│   ├── raw/                     # base PEDE original (2022–2024)
│   └── processed/               # bases geradas pela pipeline
├── src/
│   ├── harmonizacao.py          # unifica as 3 abas numa base tidy (aluno-ano)
│   ├── modelagem_base.py        # monta os pares N→N+1 com rótulos de risco
│   └── modelo.py                # treino do modelo (importado pelo app)
├── notebooks/
│   ├── 01_EDA_storytelling.ipynb  # análise exploratória (perguntas 1–8, 10, 11)
│   └── 02_modelo_risco.ipynb      # modelo preditivo (pergunta 9)
└── reports/figuras/             # figuras para a apresentação
```

## Reproduzir a pipeline

```bash
pip install -r requirements.txt
python src/harmonizacao.py        # gera data/processed/base_tidy.csv
python src/modelagem_base.py      # gera data/processed/base_modelagem.csv
streamlit run app.py              # sobe a aplicação localmente
```

## Modelo

- **Alvo:** `P(defasagem < 0 no ano N+1)` — risco de defasagem no próximo ciclo.
- **Validação:** split temporal (treina 2022→2023, testa 2023→2024); ROC-AUC ≈ 0,86.
- **Algoritmo:** HistGradientBoosting (lida com nulos nativamente; sem artefato `.pkl` — o app
  treina no startup via `@st.cache_resource`).

## Principais achados

- Na mesma coorte (468 alunos nos 3 anos), a **defasagem cai de −0,85 para −0,23**: o impacto mais
  nítido do programa é aproximar o aluno da série adequada.
- **IDA + IEG** são as maiores alavancas do INDE.
- Entre alunos **em dia**, o modelo prioriza risco com AUC ≈ 0,87 — o top 20% de maior risco cai em
  defasagem a ~3× a taxa-base.
