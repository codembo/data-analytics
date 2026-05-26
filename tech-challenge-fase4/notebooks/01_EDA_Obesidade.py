# =============================================================================
# TECH CHALLENGE — FASE 4 | FIAP POSTECH DATA ANALYTICS
# Etapa 1: Análise Exploratória de Dados (EDA)
# Dataset: obesity.csv
# =============================================================================

# %% [markdown]
# ## 1. Imports e Carregamento dos Dados

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch

# Caminho robusto — funciona de qualquer diretório
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'Obesity.csv'))

# Configuração visual
plt.rcParams['figure.facecolor'] = '#0F1117'
plt.rcParams['axes.facecolor'] = '#1A1D27'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

print(f"Shape: {df.shape}")
print(f"\nColunas: {df.columns.tolist()}")

# %% [markdown]
# ## 2. Visão Geral do Dataset

print("=" * 60)
print("INFORMAÇÕES GERAIS")
print("=" * 60)
print(f"Total de registros: {len(df)}")
print(f"Total de features: {df.shape[1] - 1}")
print(f"Variável alvo: Obesity ({df['Obesity'].nunique()} classes)")

print("\n--- Tipos de dados ---")
print(df.dtypes)

print("\n--- Valores nulos por coluna ---")
nulls = df.isnull().sum()
print(nulls[nulls >= 0])

print("\n--- Estatísticas descritivas (numéricas) ---")
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
print(df[num_cols].describe().round(2))

# %% [markdown]
# ## 3. Feature Engineering exploratório

# Criando IMC
df['BMI'] = (df['Weight'] / (df['Height'] ** 2)).round(2)

# Arredondando colunas ordinais com decimais (conforme dicionário)
for col in ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
    df[f'{col}_int'] = df[col].round().astype(int)

# Ordem lógica das classes (abaixo do peso → obesidade III)
ORDER = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I', 'Overweight_Level_II',
    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]
LABELS_PT = [
    'Peso insuficiente', 'Peso normal', 'Sobrepeso I',
    'Sobrepeso II', 'Obesidade I', 'Obesidade II', 'Obesidade III'
]
PALETTE = ['#4CAF93', '#2E9E6B', '#F5C842', '#E8973A', '#E06030', '#C03820', '#8B1A0A']
PAL_DICT = dict(zip(ORDER, PALETTE))

df['Obesity_ord'] = df['Obesity'].map({v: i for i, v in enumerate(ORDER)})

# %% [markdown]
# ## 4. Distribuição da Variável Alvo

print("\n--- Distribuição das classes ---")
counts = df['Obesity'].value_counts().reindex(ORDER)
for cls, cnt in counts.items():
    pct = cnt / len(df) * 100
    print(f"  {cls:<30} {cnt:>4}  ({pct:.1f}%)")

print(f"\nDataset {'BALANCEADO' if counts.max()/counts.min() < 1.5 else 'DESBALANCEADO'}")
print(f"  Classe maior: {counts.max()} | Classe menor: {counts.min()}")
print(f"  Razão max/min: {counts.max()/counts.min():.2f}")

# %% [markdown]
# ## 5. Visualizações — Painel 1: Visão Geral

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('EDA — Visão Geral do Dataset', fontsize=16, fontweight='bold', color='white')

title_kw = dict(color='white', fontsize=13, fontweight='bold', pad=10)

# 5.1 Distribuição das classes
ax = axes[0, 0]
counts_plot = df['Obesity'].value_counts().reindex(ORDER)
bars = ax.barh(LABELS_PT[::-1], counts_plot.values[::-1],
               color=PALETTE[::-1], edgecolor='none', height=0.65)
for bar, val in zip(bars, counts_plot.values[::-1]):
    ax.text(val + 5, bar.get_y() + bar.get_height()/2,
            str(val), va='center', color='white', fontsize=10)
ax.set_title('Distribuição das classes', **title_kw)
ax.set_xlabel('Quantidade')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# 5.2 Gênero
ax = axes[0, 1]
g_cnt = df['Gender'].value_counts()
ax.pie(g_cnt, labels=['Feminino', 'Masculino'],
       colors=['#E06030', '#4CAF93'], autopct='%1.1f%%',
       textprops={'color': 'white', 'fontsize': 11},
       startangle=90, wedgeprops={'edgecolor': '#0F1117', 'linewidth': 2})
ax.set_title('Distribuição por gênero', **title_kw)

# 5.3 IMC por classe (boxplot)
ax = axes[1, 0]
data_bmi = [df[df['Obesity'] == c]['BMI'].values for c in ORDER]
bplot = ax.boxplot(data_bmi, patch_artist=True, notch=False,
                   medianprops={'color': 'white', 'linewidth': 2},
                   whiskerprops={'color': '#aaa'}, capprops={'color': '#aaa'},
                   flierprops={'marker': 'o', 'markerfacecolor': '#555',
                               'markersize': 3, 'alpha': 0.5})
for patch, color in zip(bplot['boxes'], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.set_xticks(range(1, 8))
ax.set_xticklabels([l[:10] for l in LABELS_PT], rotation=25, ha='right', fontsize=9)
ax.set_title('IMC por classe de obesidade', **title_kw)
ax.set_ylabel('IMC (kg/m²)')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# 5.4 Scatter peso x altura
ax = axes[1, 1]
for cls, color in zip(ORDER, PALETTE):
    sub = df[df['Obesity'] == cls]
    ax.scatter(sub['Height'], sub['Weight'], c=color, s=15, alpha=0.55,
               label=cls.replace('_', ' '))
ax.set_xlabel('Altura (m)')
ax.set_ylabel('Peso (kg)')
ax.set_title('Peso × Altura por classe', **title_kw)
ax.legend(fontsize=7, facecolor='#1A1D27', labelcolor='white',
          edgecolor='#444', framealpha=0.7)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'images', 'eda_visao_geral.png'),
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("✓ Painel 1 salvo em images/eda_visao_geral.png")

# %% [markdown]
# ## 6. Visualizações — Painel 2: Hábitos e Comportamentos

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('EDA — Hábitos por Classe de Obesidade', fontsize=16,
             fontweight='bold', color='white')

# 6.1 Histórico familiar
ax = axes[0, 0]
hf = df.groupby(['Obesity', 'family_history']).size().unstack(fill_value=0).reindex(ORDER)
x = np.arange(len(ORDER))
ax.bar(x - 0.2, hf['yes'], 0.38, color='#E06030', label='Com histórico', alpha=0.9)
ax.bar(x + 0.2, hf['no'],  0.38, color='#4CAF93', label='Sem histórico', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([l[:9] for l in LABELS_PT], rotation=35, ha='right', fontsize=8)
ax.set_title('Histórico familiar', **title_kw)
ax.set_ylabel('Registros')
ax.legend(fontsize=9, facecolor='#1A1D27', labelcolor='white', edgecolor='#444')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# 6.2 Atividade física média
ax = axes[0, 1]
faf_mean = df.groupby('Obesity')['FAF'].mean().reindex(ORDER)
ax.bar(range(len(ORDER)), faf_mean.values, color=PALETTE, edgecolor='none', alpha=0.9)
for i, v in enumerate(faf_mean.values):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', color='white', fontsize=9)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([l[:9] for l in LABELS_PT], rotation=35, ha='right', fontsize=8)
ax.set_title('Atividade física média (FAF)', **title_kw)
ax.set_ylabel('FAF médio (0–3)')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# 6.3 Consumo de água
ax = axes[0, 2]
w_mean = df.groupby('Obesity')['CH2O'].mean().reindex(ORDER)
ax.bar(range(len(ORDER)), w_mean.values, color=PALETTE, edgecolor='none', alpha=0.9)
for i, v in enumerate(w_mean.values):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', color='white', fontsize=9)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([l[:9] for l in LABELS_PT], rotation=35, ha='right', fontsize=8)
ax.set_title('Consumo médio de água (CH2O)', **title_kw)
ax.set_ylabel('CH2O médio (1–3)')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# 6.4 Alimentos calóricos (%)
ax = axes[1, 0]
favc = df.groupby(['Obesity', 'FAVC']).size().unstack(fill_value=0).reindex(ORDER)
pct_yes = (favc.get('yes', 0) / favc.sum(axis=1) * 100)
ax.barh(LABELS_PT, pct_yes.values, color=PALETTE, edgecolor='none', height=0.65)
for i, v in enumerate(pct_yes.values):
    ax.text(v + 0.5, i, f'{v:.0f}%', va='center', color='white', fontsize=9)
ax.set_title('% come alimentos calóricos (FAVC)', **title_kw)
ax.set_xlabel('% de pacientes')
ax.set_xlim(0, 110)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# 6.5 Transporte
ax = axes[1, 1]
trans = df.groupby(['Obesity', 'MTRANS']).size().unstack(fill_value=0).reindex(ORDER)
pct_auto = (trans.get('Automobile', pd.Series([0]*len(ORDER))) / trans.sum(axis=1) * 100)
pct_walk = (trans.get('Walking', pd.Series([0]*len(ORDER))) / trans.sum(axis=1) * 100)
x = np.arange(len(ORDER))
ax.bar(x - 0.2, pct_auto.values, 0.38, color='#E06030', label='Automóvel', alpha=0.9)
ax.bar(x + 0.2, pct_walk.values, 0.38, color='#4CAF93', label='A pé', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([l[:9] for l in LABELS_PT], rotation=35, ha='right', fontsize=8)
ax.set_title('Transporte: automóvel vs a pé', **title_kw)
ax.set_ylabel('% de pacientes')
ax.legend(fontsize=9, facecolor='#1A1D27', labelcolor='white', edgecolor='#444')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# 6.6 Heatmap correlações
ax = axes[1, 2]
num_corr = ['Age', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Obesity_ord']
labels_corr = ['Idade', 'IMC', 'Vegetais', 'Refeições', 'Água', 'Ativ.Fís.', 'Telas', 'Obesidade']
corr = df[num_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            xticklabels=labels_corr, yticklabels=labels_corr,
            linewidths=0.3, linecolor='#333',
            annot_kws={'size': 9, 'color': 'white'})
ax.set_title('Correlações (numéricas)', **title_kw)
ax.tick_params(labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=35, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'images', 'eda_habitos.png'),
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("✓ Painel 2 salvo em images/eda_habitos.png")

# %% [markdown]
# ## 7. Insights da EDA — Resumo para a equipe médica

print("\n" + "=" * 60)
print("PRINCIPAIS INSIGHTS DA EDA")
print("=" * 60)

# 7.1 Balanceamento
print("\n[1] BALANCEAMENTO DO DATASET")
counts_pct = (df['Obesity'].value_counts().reindex(ORDER) / len(df) * 100)
for cls, pct in zip(LABELS_PT, counts_pct.values):
    print(f"    {cls:<22} {pct:.1f}%")

# 7.2 IMC por classe
print("\n[2] IMC MÉDIO POR CLASSE")
bmi_stats = df.groupby('Obesity')['BMI'].agg(['mean', 'median']).reindex(ORDER).round(1)
for cls, (mean, med) in zip(LABELS_PT, bmi_stats.values):
    print(f"    {cls:<22} média={mean:.1f} | mediana={med:.1f}")

# 7.3 Histórico familiar
print("\n[3] HISTÓRICO FAMILIAR DE SOBREPESO")
hf_pct = df.groupby('Obesity').apply(
    lambda x: (x['family_history'] == 'yes').mean() * 100
).reindex(ORDER).round(1)
for cls, pct in zip(LABELS_PT, hf_pct.values):
    print(f"    {cls:<22} {pct:.0f}% com histórico familiar")

# 7.4 Atividade física
print("\n[4] ATIVIDADE FÍSICA MÉDIA (FAF: 0=nenhuma, 3=diária)")
faf_means = df.groupby('Obesity')['FAF'].mean().reindex(ORDER).round(2)
for cls, val in zip(LABELS_PT, faf_means.values):
    print(f"    {cls:<22} FAF médio = {val:.2f}")

# 7.5 Correlações com obesidade
print("\n[5] CORRELAÇÃO COM NÍVEL DE OBESIDADE (Spearman)")
num_features = ['Age', 'BMI', 'Weight', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
corrs = df[num_features + ['Obesity_ord']].corr(method='spearman')['Obesity_ord'].drop('Obesity_ord')
corrs_sorted = corrs.abs().sort_values(ascending=False)
for feat in corrs_sorted.index:
    val = corrs[feat]
    print(f"    {feat:<12} r = {val:+.3f} {'↑ correlação positiva' if val > 0 else '↓ correlação negativa'}")

print("\n" + "=" * 60)
print("CONCLUSÕES PARA A EQUIPE MÉDICA:")
print("""
 • IMC é o preditor mais óbvio (não é surpresa — peso/altura estão no dataset).
   Para uso clínico real, os fatores comportamentais são mais acionáveis.

 • Histórico familiar: pacientes com Obesidade Tipo III têm ~90%+ de histórico
   familiar positivo — triagem familiar é recomendada.

 • Atividade física: quem tem Obesidade I-III pratica MENOS atividade que
   pessoas com peso normal — intervenção focada em exercício é prioritária.

 • Consumo de alimentos calóricos (FAVC=sim) cresce com o nível de obesidade.

 • Uso de telas (TUE) tem correlação positiva fraca com obesidade.

 • Transporte ativo (a pé) é mais comum em classes de peso mais baixo.
""")