# 🏥 Tech Challenge — Fase 4 | Data Analytics
### Sistema Preditivo de Obesidade

**FIAP PósTech — Data Analytics**

**Grupo:**
- Misael Oliveira
- Gustavo Bacelar Horita
- Álvaro de Freitas Pinto
- Victor Fernando Gil

---

## 📌 Sobre o projeto

Desenvolvido como parte do Tech Challenge da Fase 4 do curso de Data Analytics da FIAP PósTech, este projeto consiste em um sistema preditivo para auxiliar a equipe médica a diagnosticar o nível de obesidade de pacientes com base em dados comportamentais, físicos e histórico familiar.

---

## 🔗 Links

| Recurso | Link |
|---|---|
| 🚀 App Streamlit (deploy) | https://data-analytics-fase4.streamlit.app/ |
| 📁 Repositório GitHub | https://github.com/codembo/data-analytics |

---

## 🎯 Resultados do modelo

| Métrica | Valor |
|---|---|
| Algoritmo | GradientBoostingClassifier |
| Acurácia (teste) | **95.7%** |
| F1-score macro | **95.6%** |
| Meta FIAP (>75%) | ✅ Atingida |

---

## 🗂️ Estrutura do projeto

```
tech-challenge-fase4/
├── app.py                          # App Streamlit (preditivo + dashboard)
├── requirements.txt                # Dependências
├── data/
│   └── Obesity.csv                 # Dataset original
├── models/
│   ├── modelo_obesidade.pkl        # Modelo treinado (GBM)
│   ├── scaler.pkl                  # StandardScaler treinado
│   └── metadata.json               # Mapeamentos e metadados
├── notebooks/
│   ├── 01_EDA_Obesidade.py         # Análise exploratória
│   ├── 02_Feature_Engineering.py   # Pipeline de transformação
│   └── 03_Modelo_ML.py             # Treinamento e avaliação
└── images/
    ├── eda_visao_geral.png
    ├── eda_habitos.png
    ├── feature_engineering.png
    └── modelo_ml.png
```

---

## 🔬 Pipeline de Machine Learning

### Etapa 1 — Análise Exploratória (EDA)
- Distribuição das 7 classes de obesidade (dataset balanceado, razão max/min = 1.29)
- Análise de correlações entre hábitos e nível de obesidade
- Visualizações: boxplot de IMC, scatter peso×altura, histogramas de hábitos

### Etapa 2 — Feature Engineering
| Transformação | Colunas |
|---|---|
| Arredondamento (decimais → inteiros) | FCVC, NCP, CH2O, FAF, TUE |
| Label Encoding binário (0/1) | Gender, family_history, FAVC, SMOKE, SCC |
| Ordinal Encoding (frequência 0–3) | CAEC, CALC |
| One-Hot Encoding | MTRANS (5 categorias) |
| StandardScaler (fit só no treino) | Todas as features numéricas |

### Etapa 3 — Treinamento e Seleção do Modelo

| Modelo | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 86.7% | 87.7% |
| Random Forest | 94.2% | 93.9% |
| **Gradient Boosting** | **96.3%** | **95.7%** ← escolhido |
| SVM (RBF) | 85.5% | 86.1% |

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

### Etapa 4 — Deploy (Streamlit)
- **Aba 1 — Sistema Preditivo:** formulário com dados do paciente, resultado do diagnóstico, probabilidade por classe e interpretação clínica
- **Aba 2 — Dashboard Analítico:** 5 KPIs, 6 gráficos interativos (Plotly) e 6 insights para a equipe médica

---

## 📊 Principais insights para a equipe médica

1. **Histórico familiar** é o segundo preditor mais forte — 90%+ dos pacientes com Obesidade Tipo III têm histórico familiar positivo
2. **Sedentarismo** acompanha a progressão da obesidade — FAF médio cai de 1.25 (peso normal) para 0.64 (Obesidade III)
3. **Alimentação calórica** (FAVC e CAEC) cresce proporcionalmente ao nível de obesidade
4. **Transporte ativo** (caminhar) é mais comum em pessoas com peso normal
5. **Hidratação** e **monitoramento calórico** estão associados a menores níveis de obesidade

---

## ⚙️ Como executar localmente

```bash
# 1. Clone o repositório
git clone https://github.com/codembo/data-analytics.git
cd data-analytics/tech-challenge-fase4

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute os notebooks (opcional — modelos já estão em models/)
python3 notebooks/01_EDA_Obesidade.py
python3 notebooks/02_Feature_Engineering.py
python3 notebooks/03_Modelo_ML.py

# 4. Rode o app
streamlit run app.py
```

---

## 🛠️ Tecnologias utilizadas

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.18-purple)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)

---

*FIAP PósTech — Data Analytics | Tech Challenge Fase 4 | 2026*