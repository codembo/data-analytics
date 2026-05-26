# =============================================================================
# TECH CHALLENGE — FASE 4 | FIAP POSTECH DATA ANALYTICS
# Etapa 2: Feature Engineering + Preparação do Pipeline
# Dataset: obesity.csv
# =============================================================================

# %% [markdown]
# ## 1. Imports

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'Obesity.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
IMAGES_DIR  = os.path.join(BASE_DIR, 'images')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Configuração visual
plt.rcParams['figure.facecolor'] = '#0F1117'
plt.rcParams['axes.facecolor']   = '#1A1D27'
plt.rcParams['text.color']       = 'white'
plt.rcParams['axes.labelcolor']  = 'white'
plt.rcParams['xtick.color']      = 'white'
plt.rcParams['ytick.color']      = 'white'

# %% [markdown]
# ## 2. Carregamento e cópia de segurança

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()
print(f"Dataset carregado: {df.shape}")

# %% [markdown]
# ## 3. Ordem lógica das classes (usada em todo o projeto)

ORDER = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I',  'Overweight_Level_II',
    'Obesity_Type_I',      'Obesity_Type_II', 'Obesity_Type_III'
]
LABELS_PT = [
    'Peso insuficiente', 'Peso normal', 'Sobrepeso I',
    'Sobrepeso II', 'Obesidade I', 'Obesidade II', 'Obesidade III'
]

# %% [markdown]
# ## 4. Arredondamento de colunas ordinais com decimais
#
# Conforme o dicionário de dados, FCVC, NCP, CH2O, FAF e TUE são escalas
# inteiras (1–3 ou 0–3), mas o dataset contém ruído decimal. Arredondamos.

ROUND_COLS = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
print("\n--- Antes do arredondamento ---")
print(df[ROUND_COLS].describe().round(3))

for col in ROUND_COLS:
    df[col] = df[col].round().astype(int)

print("\n--- Após arredondamento ---")
print(df[ROUND_COLS].describe())

# %% [markdown]
# ## 5. Encoding das variáveis categóricas
#
# Estratégia:
#   - Binárias (yes/no, Female/Male)  → Label Encoding manual (0/1)
#   - Ordinais (CAEC, CALC)           → Ordinal Encoding com ordem de frequência
#   - Nominais (MTRANS)               → One-Hot Encoding

# 5.1 Binárias
print("\n--- Encoding binário ---")
binary_map = {'yes': 1, 'no': 0, 'Female': 0, 'Male': 1}
BINARY_COLS = ['Gender', 'family_history', 'FAVC', 'SMOKE', 'SCC']
for col in BINARY_COLS:
    df[col] = df[col].map(binary_map)
    print(f"  {col}: {df[col].unique()}")

# 5.2 Ordinais
print("\n--- Encoding ordinal ---")
# CAEC e CALC têm a mesma escala de frequência
freq_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
for col in ['CAEC', 'CALC']:
    df[col] = df[col].map(freq_map)
    print(f"  {col}: {sorted(df[col].unique())}")

# 5.3 Nominal — MTRANS (One-Hot)
print("\n--- One-Hot Encoding: MTRANS ---")
df = pd.get_dummies(df, columns=['MTRANS'], drop_first=False, dtype=int)
mtrans_cols = [c for c in df.columns if c.startswith('MTRANS_')]
print(f"  Colunas geradas: {mtrans_cols}")

# %% [markdown]
# ## 6. Encoding da variável alvo

target_map = {v: i for i, v in enumerate(ORDER)}
reverse_map = {i: v for i, v in enumerate(ORDER)}   # para decodificar predições

df['Obesity_encoded'] = df['Obesity'].map(target_map)

print("\n--- Mapeamento da variável alvo ---")
for k, v in target_map.items():
    print(f"  {v} → {k}")

# %% [markdown]
# ## 7. Definição de features e target

FEATURE_COLS = [c for c in df.columns if c not in ['Obesity', 'Obesity_encoded']]
TARGET_COL   = 'Obesity_encoded'

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")
print(f"\nX shape: {X.shape}")
print(f"y distribuição:\n{y.value_counts().sort_index()}")

# %% [markdown]
# ## 8. Divisão treino/teste estratificada
#
# stratify=y garante que a proporção das 7 classes seja preservada
# em treino e teste.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n--- Split treino/teste ---")
print(f"  Treino : {X_train.shape[0]} registros ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Teste  : {X_test.shape[0]} registros ({X_test.shape[0]/len(X)*100:.1f}%)")

print("\n--- Proporção das classes no treino ---")
for i, label in enumerate(LABELS_PT):
    n = (y_train == i).sum()
    print(f"  {label:<22} {n:>4} ({n/len(y_train)*100:.1f}%)")

# %% [markdown]
# ## 9. Normalização com StandardScaler
#
# Importante: fit apenas no treino, transform em treino e teste.
# Isso evita data leakage.

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n--- Verificação do scaler (primeiras 3 features) ---")
print(f"  Média treino  (após scale): {X_train_sc[:, :3].mean(axis=0).round(4)}")
print(f"  Desvio treino (após scale): {X_train_sc[:, :3].std(axis=0).round(4)}")

# Reconvertendo para DataFrame para facilitar análise
X_train_sc_df = pd.DataFrame(X_train_sc, columns=FEATURE_COLS)
X_test_sc_df  = pd.DataFrame(X_test_sc,  columns=FEATURE_COLS)

# %% [markdown]
# ## 10. Visualização pós-encoding

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Feature Engineering — Distribuições pós-encoding',
             fontsize=14, fontweight='bold', color='white')

title_kw = dict(color='white', fontsize=12, fontweight='bold', pad=8)
PALETTE = ['#4CAF93','#2E9E6B','#F5C842','#E8973A','#E06030','#C03820','#8B1A0A']

# 10.1 Distribuição das features numéricas normalizadas
ax = axes[0]
X_train_sc_df[['Age','Weight','Height','FAF','TUE']].plot(
    kind='box', ax=ax, patch_artist=True,
    medianprops={'color':'white','linewidth':2},
    whiskerprops={'color':'#aaa'}, capprops={'color':'#aaa'},
    flierprops={'marker':'o','markerfacecolor':'#555','markersize':3}
)
boxes = ax.patches
colors = ['#4CAF93','#E06030','#F5C842','#E8973A','#C03820']
for patch, color in zip(boxes, colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_title('Features numéricas (normalizadas)', **title_kw)
ax.set_ylabel('Valor padronizado (z-score)')
ax.tick_params(colors='white')
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# 10.2 Correlação das features com o target
ax = axes[1]
corr_target = pd.concat([X_train_sc_df, y_train.reset_index(drop=True)], axis=1)\
    .corr()['Obesity_encoded'].drop('Obesity_encoded').abs().sort_values(ascending=True)
colors_corr = ['#E06030' if v > 0.4 else '#F5C842' if v > 0.2 else '#4CAF93'
               for v in corr_target.values]
ax.barh(corr_target.index, corr_target.values, color=colors_corr, edgecolor='none')
ax.set_title('Correlação |r| com obesidade', **title_kw)
ax.set_xlabel('|Correlação de Pearson|')
ax.tick_params(colors='white', labelsize=8)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color('#444')
ax.spines['left'].set_color('#444')

# 10.3 Distribuição do target no treino
ax = axes[2]
counts = y_train.value_counts().sort_index()
bars = ax.bar(range(len(ORDER)), counts.values, color=PALETTE, edgecolor='none')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            str(val), ha='center', color='white', fontsize=9)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([l[:10] for l in LABELS_PT], rotation=30, ha='right', fontsize=8)
ax.set_title('Distribuição do target (treino)', **title_kw)
ax.set_ylabel('Registros')
ax.tick_params(colors='white')
for sp in ['top','right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'feature_engineering.png'),
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("✓ Gráfico salvo em images/feature_engineering.png")

# %% [markdown]
# ## 11. Salvando artefatos para a Etapa 3

# Scaler — será reaproveitado no app Streamlit
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print(f"\n✓ Scaler salvo em models/scaler.pkl")

# DataFrames prontos para modelagem
X_train_sc_df.to_csv(os.path.join(MODELS_DIR, 'X_train.csv'), index=False)
X_test_sc_df.to_csv(os.path.join(MODELS_DIR,  'X_test.csv'),  index=False)
y_train.reset_index(drop=True).to_csv(os.path.join(MODELS_DIR, 'y_train.csv'), index=False)
y_test.reset_index(drop=True).to_csv(os.path.join(MODELS_DIR,  'y_test.csv'),  index=False)
print("✓ Datasets de treino/teste salvos em models/")

# Metadados úteis para o app
import json
meta = {
    'feature_cols':  FEATURE_COLS,
    'target_map':    target_map,
    'reverse_map':   {str(k): v for k, v in reverse_map.items()},
    'order':         ORDER,
    'labels_pt':     LABELS_PT,
    'binary_map':    {'yes': 1, 'no': 0, 'Female': 0, 'Male': 1},
    'freq_map':      {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'mtrans_cols':   mtrans_cols,
    'round_cols':    ROUND_COLS,
}
with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("✓ Metadados salvos em models/metadata.json")

# %% [markdown]
# ## 12. Resumo do pipeline

print("\n" + "=" * 60)
print("RESUMO DO FEATURE ENGINEERING")
print("=" * 60)
print(f"""
Transformações aplicadas:
  1. Arredondamento : FCVC, NCP, CH2O, FAF, TUE → inteiros
  2. Encoding binário : Gender, family_history, FAVC, SMOKE, SCC → 0/1
  3. Encoding ordinal : CAEC, CALC → escala 0-3 (frequência)
  4. One-Hot Encoding : MTRANS → 5 colunas binárias
  5. Target encoding  : Obesity → 0 a 6 (ordem clínica)
  6. Normalização     : StandardScaler (fit só no treino)

Dataset resultante:
  Features : {len(FEATURE_COLS)} colunas
  Treino   : {X_train.shape[0]} registros
  Teste    : {X_test.shape[0]} registros

Artefatos gerados (pasta models/):
  scaler.pkl       → scaler treinado (reusar no Streamlit)
  X_train.csv      → features de treino normalizadas
  X_test.csv       → features de teste normalizadas
  y_train.csv      → target de treino
  y_test.csv       → target de teste
  metadata.json    → mapeamentos e listas de features
""")