<div align="center">

# 🏥 Sistema Preditivo de Obesidade

**FIAP PósTech · Data Analytics | Tech Challenge Fase 4**

[![App ao vivo](https://img.shields.io/badge/🚀_App_Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://data-analytics-fase4.streamlit.app/)
[![Acurácia](https://img.shields.io/badge/Acurácia-95.7%25-4CAF50?style=for-the-badge)]()
[![F1 Macro](https://img.shields.io/badge/F1--macro-95.6%25-4CAF50?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]()

</div>

---

## 📌 Sobre o projeto

Sistema de apoio à decisão clínica desenvolvido para o Tech Challenge da Fase 4 do curso de **Data Analytics da FIAP PósTech**. O objetivo é auxiliar equipes médicas a prever o nível de obesidade de pacientes com base em dados comportamentais, físicos e histórico familiar.

> ⚕️ **Aviso clínico:** Esta ferramenta é um sistema de apoio à decisão e não substitui a avaliação do profissional de saúde.

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
| 🚀 App Streamlit (sistema preditivo + dashboard) | https://data-analytics-fase4.streamlit.app/ |
| 📊 Dashboard analítico | https://data-analytics-fase4.streamlit.app/ → aba "Dashboard Analítico" |
| 💻 Repositório GitHub | https://github.com/codembo/data-analytics |

---

## 🎯 Resultados

| Métrica | Valor |
|---|---|
| Algoritmo | GradientBoostingClassifier |
| Acurácia no teste | **95.7%** |
| F1-score macro | **95.6%** |
| Validação cruzada (5-fold) | **96.3% ± 0.3%** |
| Meta FIAP (>75%) | ✅ Superada em +20.7pp |
| Fairness (Feminino vs Masculino) | ✅ Δ 2.5pp, dentro do limite aceitável |

---

## 🗂️ Estrutura do projeto

```
tech-challenge-fase4/
├── app.py                          # App Streamlit (3 abas)
├── requirements.txt                # Dependências com versões fixas
├── data/
│   └── Obesity.csv                 # Dataset original (2.111 registros)
├── models/
│   ├── modelo_obesidade.pkl        # Modelo GBM treinado (todas as features)
│   ├── modelo_comportamental.pkl   # Modelo GBM sem peso/altura (triagem preventiva)
│   ├── scaler.pkl                  # StandardScaler treinado no conjunto de treino
│   └── metadata.json               # Mapeamentos, features e métricas
├── notebooks/
│   ├── 01_EDA_Obesidade.py         # Análise exploratória completa
│   ├── 02_Feature_Engineering.py   # Pipeline de transformação
│   ├── 03_Modelo_ML.py             # Treinamento e avaliação de 4 modelos
│   └── 04_Analise_Avancada.py      # Modelo comportamental + fairness + limitações
└── images/
    ├── eda_visao_geral.png
    ├── eda_habitos.png
    ├── feature_engineering.png
    ├── modelo_ml.png
    └── analise_avancada.png
```

---

## 🔬 Pipeline de Machine Learning

### Etapa 1: Análise Exploratória (EDA)
- Dataset balanceado: 7 classes com distribuição entre 12–17% cada (razão max/min = 1.29)
- Zero valores nulos em todas as 17 colunas
- Insights: histórico familiar positivo em 90%+ dos pacientes com Obesidade Tipo III; FAF médio cai de 1.25 (peso normal) para 0.64 (Obesidade III)

### Etapa 2: Feature Engineering

| Transformação | Colunas envolvidas |
|---|---|
| Arredondamento de decimais para inteiros | FCVC, NCP, CH2O, FAF, TUE |
| Label Encoding binário (0/1) | Gender, family_history, FAVC, SMOKE, SCC |
| Ordinal Encoding (frequência 0–3) | CAEC, CALC |
| One-Hot Encoding | MTRANS → 5 colunas |
| StandardScaler (fit apenas no treino) | Todas as features numéricas |
| Target Encoding | Obesity → 0 a 6 (ordem clínica crescente) |

**Resultado:** 20 features | 1.688 registros treino | 423 registros teste

### Etapa 3: Comparação de modelos

| Modelo | CV Accuracy | Test Accuracy | F1-macro |
|---|---|---|---|
| Logistic Regression | 86.7% | 87.7% | 87.3% |
| Random Forest | 94.2% | 93.9% | 93.7% |
| **Gradient Boosting** | **96.3%** | **95.7%** | **95.6%** |
| SVM (RBF) | 85.5% | 86.1% | 85.8% |

**Hiperparâmetros do modelo final:**
```python
GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
```

### Etapa 4: Análise avançada

**Modelo comportamental (sem peso e altura):**
Treinado para validar que os hábitos de vida têm poder preditivo real independente do IMC.
- Acurácia: **79.0%**, supera a meta de 75% apenas com dados comportamentais
- Aplicação clínica: triagem preventiva quando exame físico ainda não foi realizado

**Análise de fairness:**
- Feminino (n=201): 97.5% | Masculino (n=222): 95.0% | Δ = 2.5pp ✅

---

## 📊 O app Streamlit tem 3 abas

**🔬 Sistema Preditivo**
- Formulário completo com dados do paciente
- Resultado com diagnóstico, confiança e orientação clínica
- Lista automática de fatores de risco identificados
- Modo comportamental (toggle) para triagem sem peso/altura

**📊 Dashboard Analítico**
- Filtros por gênero, faixa etária e classe de obesidade
- 5 KPIs com contexto comparativo (incluindo média nacional IBGE)
- 6 gráficos interativos com Plotly
- 6 insights em linguagem de negócio para a equipe médica

**🧠 Transparência do Modelo**
- Comparativo modelo completo vs comportamental
- Análise de fairness documentada
- 5 limitações conhecidas explicitadas
- Parâmetros e métricas de treinamento

---

## 💡 Principais insights para a equipe médica

1. **Histórico familiar:** 90%+ dos pacientes com Obesidade Tipo III têm histórico familiar positivo, triagem preventiva em familiares é recomendada
2. **Sedentarismo:** FAF médio cai consistentemente com o aumento da obesidade, intervenção de exercício é a ação mais impactante
3. **Alimentação:** Consumo de alimentos calóricos (FAVC) e lanches entre refeições (CAEC) crescem proporcionalmente ao nível de obesidade
4. **Prevalência:** 46% dos pacientes do dataset têm obesidade, proporção acima da média nacional de 22% (IBGE 2023)
5. **Transporte:** Uso de automóvel é mais prevalente em pacientes obesos; caminhar é mais comum em pessoas com peso normal

---

## ⚙️ Como executar localmente

```bash
# 1. Clone o repositório
git clone https://github.com/codembo/data-analytics.git
cd data-analytics/tech-challenge-fase4

# 2. Instale as dependências
pip install -r requirements.txt

# 3. (Opcional) Rode os notebooks em ordem
python3 notebooks/01_EDA_Obesidade.py
python3 notebooks/02_Feature_Engineering.py
python3 notebooks/03_Modelo_ML.py
python3 notebooks/04_Analise_Avancada.py

# 4. Suba o app
streamlit run app.py
```

> Os modelos já estão pré-treinados em `models/`. Os passos 3 são opcionais, servem para reproduzir o pipeline completo.

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
<sub>FIAP PósTech · Data Analytics · Tech Challenge Fase 4 · 2026</sub><br>
<sub>⚕️ Ferramenta de apoio à decisão clínica, não substitui avaliação médica profissional</sub>
</div>
