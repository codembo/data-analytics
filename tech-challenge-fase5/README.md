<div align="center">

# 🎓 Sistema de Alerta Precoce de Defasagem

**FIAP PósTech — Data Analytics | Tech Challenge Fase 5 — Datathon Passos Mágicos**

[![App ao vivo](https://img.shields.io/badge/🚀_App_Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://data-analytics-tech-challenge-fase5.streamlit.app/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.86-4CAF50?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]()

</div>

---

## 📌 Sobre o projeto

Datathon da Fase 5 do curso de **Data Analytics da FIAP PósTech**, com dados reais da
**Associação Passos Mágicos**. A base **PEDE** (Pesquisa Extensiva do Desenvolvimento Educacional)
traz um retrato anual de cada aluno (2022–2024). Usamos o `RA` para ligar o mesmo aluno entre anos e
construir um problema **longitudinal**: a partir do retrato do aluno no ano N, prever se ele estará
**defasado no ano N+1** — permitindo à ONG priorizar a intervenção *antes* da queda.

> 🎓 **Aviso:** Esta ferramenta é um sistema de apoio à decisão pedagógica e não substitui a
> avaliação da equipe da ONG.

---

## 👥 Grupo

| Nome |
|---|
| Misael Oliveira |
| Gustavo Bacelar Horita |
| Álvaro de Freitas Pinto |
| Victor Fernando Gil |

---

## 🔗 Links de entrega

| Recurso | Link |
|---|---|
| 🚀 App Streamlit (predição + priorização + transparência) | https://data-analytics-tech-challenge-fase5.streamlit.app/ |
| 💻 Repositório GitHub | https://github.com/codembo/data-analytics |

---

## 🎯 Resultados

| Métrica | Valor |
|---|---|
| Algoritmo | HistGradientBoostingClassifier |
| ROC-AUC — teste temporal (2023→2024) | **0.859** |
| PR-AUC — teste temporal | **0.814** |
| ROC-AUC — validação cruzada (5-fold) | **0.908 ± 0.010** |
| ROC-AUC — subgrupo "hoje em dia" (o alvo real) | **0.867** |
| Lift do top de risco vs. taxa-base | **≈ 3×** |
| Recall no corte operacional (0.4) | **81%** (precisão 68%) |

---

## 🗂️ Estrutura do projeto

```
tech-challenge-fase5/
├── app.py                          # App Streamlit (3 abas)
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── PEDE_PASSOS_DESAFIO.xlsx    # Base bruta original (abas 2022/2023/2024)
│   └── processed/
│       ├── base_tidy.csv               # 1 linha por aluno-ano (3.030 registros)
│       └── base_modelagem.csv          # 1 linha por par N→N+1 (1.365 pares)
├── src/
│   ├── harmonizacao.py             # Unifica as 3 abas (esquemas divergentes) numa base tidy
│   ├── modelagem_base.py           # Monta os pares N→N+1 e deriva os rótulos de risco
│   └── modelo.py                   # Treino do modelo (importado pelo notebook e pelo app)
├── notebooks/
│   ├── 01_EDA_storytelling.ipynb   # Perguntas de negócio 1–8, 10 e 11
│   └── 02_modelo_risco.ipynb       # Modelo preditivo de risco (pergunta 9)
└── reports/
    └── figuras/                    # Figuras exportadas para a apresentação
```

---

## 🔬 Pipeline

### Etapa 1 — Harmonização
`src/harmonizacao.py` lê as 3 abas do PEDE (2022–2024), que têm **esquemas divergentes**
(colunas renomeadas, formatos de fase distintos — "3", "FASE 3", "3A"), e as unifica numa base
tidy: uma linha por (aluno, ano). O `RA` é a chave estável que liga o mesmo aluno ao longo do tempo.
**Resultado:** 3.030 registros aluno-ano.

### Etapa 2 — Análise Exploratória (EDA)
`notebooks/01_EDA_storytelling.ipynb` responde as perguntas de negócio 1–8, 10 e 11:

| Pergunta | Achado |
|---|---|
| Q1 — Adequação ao nível | Adequados/adiantados sobem de 30,1% (2022) para 53,8% (2024); casos severos praticamente eliminados |
| Q2 — Desempenho (IDA) | Vale na fase 3 (Fund. II) — transição para a adolescência é o ponto mais sensível |
| Q3/Q4 — Engajamento vs. autoavaliação | IEG correlaciona com IDA e IPV (r=0,54); IAA descola da performance real (r≈0,22) |
| Q5 — Psicossocial (IPS) | Sem evidência de que antecede quedas — dimensão ortogonal às demais |
| Q6/Q7 — Psicopedagógico e Ponto de Virada | IPV é puxado por IPP (0,61), IEG (0,56) e IDA (0,56) |
| Q8 — Alavancas do INDE | IDA (0,42) e IEG (0,35) têm o maior retorno marginal |
| Q10 — Efetividade (mesma coorte, 468 alunos) | Defasagem melhora de −0,85 para −0,23; INDE fica estável — fechar o gap série-idade é o impacto mais nítido do programa |
| Q11 — Insight próprio | Existe um subgrupo previsível de risco — porta de entrada para o modelo (Q9) |

### Etapa 3 — Base de modelagem
`src/modelagem_base.py` une, pelo `RA`, as features do aluno no ano N aos desfechos do ano N+1,
operacionalizando o problema como **sistema de alerta precoce**. Deriva o rótulo
`y_risco = (defasagem_next < 0)`. **Resultado:** 1.365 pares N→N+1.

### Etapa 4 — Modelo preditivo (pergunta 9)
`notebooks/02_modelo_risco.ipynb` / `src/modelo.py`:
- **Split temporal** (não aleatório): treina em pares 2022→2023 (600), testa em 2023→2024 (765) —
  simula o uso real de prever uma coorte futura, sob deslocamento real de prevalência (61% → 40%).
- **Comparação de modelos:**

  | Modelo | ROC-AUC | PR-AUC |
  |---|---|---|
  | Baseline | 0.500 | 0.403 |
  | Regressão Logística | 0.808 | 0.761 |
  | **HistGradientBoosting** | **0.859** | **0.814** |

- **HistGradientBoostingClassifier** foi escolhido por lidar nativamente com nulos (indicadores
  como IDA/IEG faltam para parte dos alunos) sem precisar de imputação.
- **Calibração e limiar de alerta:** corte operacional em 0.4 cobre 81% dos casos reais de risco com
  68% de precisão — equilíbrio adequado para priorizar com recursos limitados.
- **O resultado que importa:** restrito só a alunos **hoje em dia** (o cenário real de uso — prevenir
  quem ainda não caiu), o modelo alcança AUC ≈ 0.867 e o quintil de maior risco cai ~3× mais que a
  taxa-base.
- **Interpretabilidade:** posição estrutural (fase, defasagem atual, idade) domina a previsão; entre
  os indicadores comportamentais, o **IPV (ponto de virada)** lidera. Gênero é irrelevante
  (importância 0.001) — sem discriminação por sexo.

**Hiperparâmetros do modelo final:**
```python
HistGradientBoostingClassifier(
    random_state=42, max_iter=300, learning_rate=0.05, max_depth=4
)
```

> O app **não usa artefato `.pkl`** — treina o modelo no startup via `@st.cache_resource`, padrão
> que elimina incompatibilidades de versão do scikit-learn no deploy (lição aprendida na Fase 4).

---

## 📊 O app Streamlit tem 3 abas

**🔎 Predição individual**
- Formulário com o retrato atual de um aluno (fase, defasagem, indicadores)
- Probabilidade de defasagem no próximo ciclo, com faixa de risco (baixo/médio/alto) e orientação

**📋 Priorização da turma**
- Pontua toda a coorte de 2024 pelo risco previsto para o próximo ciclo
- Destaque para alunos **hoje em dia mas em risco** — onde a intervenção precoce mais rende
- Lista ordenável e exportável em CSV para a equipe da ONG

**ℹ️ Sobre o modelo**
- Métricas de desempenho (ROC-AUC temporal, pares de treino)
- Explicação de como o modelo funciona, o que pesa na previsão e seus limites
- Importância das variáveis (permutation importance) sob demanda

---

## 💡 Principais achados

1. **O programa fecha o gap série-idade.** Na mesma coorte (468 alunos nos 3 anos), a defasagem
   melhora de −0,85 para −0,23 — o impacto mais nítido e comprovado do programa.
2. **Engajamento e desempenho andam juntos**; autoavaliação e aspecto psicossocial são dimensões
   à parte, pouco correlacionadas com o desempenho real.
3. **IDA + IEG são as maiores alavancas do INDE** — maior retorno marginal está em desempenho
   acadêmico e engajamento, não em autoavaliação.
4. **Existe um subgrupo previsível de risco.** Mesmo entre alunos hoje em dia, o modelo separa bem
   quem tende a cair — priorizar o grupo de maior risco previsto rende ~3× a taxa-base de acerto.
5. **O modelo não discrimina por gênero** — importância de gênero na previsão é praticamente nula
   (0.001), um ponto relevante de equidade para uma ONG.

---

## ⚙️ Como executar localmente

```bash
# 1. Clone o repositório
git clone https://github.com/codembo/data-analytics.git
cd data-analytics/tech-challenge-fase5

# 2. Instale as dependências
pip install -r requirements.txt

# 3. (Opcional) Reproduza a pipeline de dados
python3 src/harmonizacao.py       # gera data/processed/base_tidy.csv
python3 src/modelagem_base.py     # gera data/processed/base_modelagem.csv

# 4. Suba o app
streamlit run app.py
```

> As bases processadas já estão versionadas em `data/processed/`. O passo 3 é opcional — serve para
> reproduzir a pipeline completa a partir da base bruta. O modelo não tem artefato `.pkl`: é treinado
> no startup do app.

---

## 🛠️ Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

<div align="center">
<sub>FIAP PósTech · Data Analytics · Tech Challenge Fase 5 · Datathon Passos Mágicos · 2026</sub><br>
<sub>🎓 Ferramenta de apoio à decisão pedagógica — não substitui a avaliação da equipe da ONG</sub>
</div>
