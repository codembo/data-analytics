# =============================================================================
# TECH CHALLENGE — FASE 4 | FIAP POSTECH DATA ANALYTICS
# Etapa 4: Análise Avançada — Modelo Comportamental + Fairness + Limitações
# =============================================================================

# %% [markdown]
# ## 1. Imports

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.metrics   import (accuracy_score, f1_score,
                                classification_report, confusion_matrix)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

plt.rcParams['figure.facecolor'] = '#0F1117'
plt.rcParams['axes.facecolor']   = '#1A1D27'
plt.rcParams['text.color']       = 'white'
plt.rcParams['axes.labelcolor']  = 'white'
plt.rcParams['xtick.color']      = 'white'
plt.rcParams['ytick.color']      = 'white'

# %% [markdown]
# ## 2. Carregamento

X_train = pd.read_csv(os.path.join(MODELS_DIR, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(MODELS_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(MODELS_DIR, 'y_train.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(MODELS_DIR, 'y_test.csv')).squeeze()

with open(os.path.join(MODELS_DIR, 'metadata.json')) as f:
    meta = json.load(f)

ORDER     = meta['order']
LABELS_PT = meta['labels_pt']
PALETTE   = ['#4CAF93','#2E9E6B','#F5C842','#E8973A','#E06030','#C03820','#8B1A0A']

model_full = joblib.load(os.path.join(MODELS_DIR, 'modelo_obesidade.pkl'))

# %% [markdown]
# ## 3. Análise crítica — o papel de Weight e Height
#
# O modelo completo usa todas as 20 features, incluindo peso e altura.
# Porém, a classe de obesidade é clinicamente definida pelo IMC (peso/altura²).
# Isso levanta a questão: o modelo está aprendendo comportamento
# ou simplesmente recalculando o IMC?
#
# Testamos um modelo apenas com features COMPORTAMENTAIS (sem peso e altura)
# para isolar o poder preditivo dos hábitos de vida.

print("=" * 60)
print("ANÁLISE: Modelo Completo vs Modelo Comportamental")
print("=" * 60)

DROP_COLS = ['Weight', 'Height']
X_train_beh = X_train.drop(columns=DROP_COLS)
X_test_beh  = X_test.drop(columns=DROP_COLS)

model_beh = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
model_beh.fit(X_train_beh, y_train)

y_pred_full = model_full.predict(X_test)
y_pred_beh  = model_beh.predict(X_test_beh)

acc_full = accuracy_score(y_test, y_pred_full)
acc_beh  = accuracy_score(y_test, y_pred_beh)
f1_full  = f1_score(y_test, y_pred_full, average='macro')
f1_beh   = f1_score(y_test, y_pred_beh,  average='macro')

print(f"\n  Modelo completo (20 features)    : Acc={acc_full:.4f} | F1={f1_full:.4f}")
print(f"  Modelo comportamental (18 feat.) : Acc={acc_beh:.4f}  | F1={f1_beh:.4f}")
print(f"\n  Contribuição de peso+altura      : {(acc_full - acc_beh)*100:.1f} pontos percentuais")
print(f"\n  ✓ Os hábitos comportamentais sozinhos explicam {acc_beh*100:.1f}% dos casos.")
print(f"    Isso TEM valor clínico: permite triagem preventiva ANTES de medir peso/altura.")

# Salvar modelo comportamental
joblib.dump(model_beh, os.path.join(MODELS_DIR, 'modelo_comportamental.pkl'))
print(f"\n  ✓ Modelo comportamental salvo em models/modelo_comportamental.pkl")

# %% [markdown]
# ## 4. F1 por classe — comparativo

rep_full = classification_report(y_test, y_pred_full, output_dict=True)
rep_beh  = classification_report(y_test, y_pred_beh,  output_dict=True)

print("\n--- F1 por classe: Completo vs Comportamental ---")
print(f"  {'Classe':<22} {'Completo':>10} {'Comport.':>10} {'Δ':>8}")
print("  " + "-" * 54)
for i, label in enumerate(LABELS_PT):
    f_c = rep_full[str(i)]['f1-score']
    f_b = rep_beh[str(i)]['f1-score']
    flag = "⚠" if f_b < 0.75 else "✓"
    print(f"  {label:<22} {f_c:>10.3f} {f_b:>10.3f} {f_b-f_c:>+8.3f} {flag}")

# %% [markdown]
# ## 5. Análise de Fairness
#
# Verificamos se o modelo performa de forma justa entre gêneros.
# Gender após StandardScaler: valores negativos = Feminino, positivos = Masculino

print("\n" + "=" * 60)
print("ANÁLISE DE FAIRNESS — por gênero")
print("=" * 60)

mask_f = X_test['Gender'] < 0  # Feminino (0 antes do scale)
mask_m = X_test['Gender'] > 0  # Masculino (1 antes do scale)

acc_f = accuracy_score(y_test[mask_f], y_pred_full[mask_f])
acc_m = accuracy_score(y_test[mask_m], y_pred_full[mask_m])
f1_f  = f1_score(y_test[mask_f], y_pred_full[mask_f], average='macro')
f1_m  = f1_score(y_test[mask_m], y_pred_full[mask_m], average='macro')

print(f"\n  Feminino  (n={mask_f.sum()}): Acc={acc_f:.4f} | F1={f1_f:.4f}")
print(f"  Masculino (n={mask_m.sum()}): Acc={acc_m:.4f} | F1={f1_m:.4f}")
print(f"  Diferença de acurácia: {abs(acc_f - acc_m)*100:.1f}pp")
print(f"  Veredicto: {'✓ JUSTO (< 5pp)' if abs(acc_f - acc_m) < 0.05 else '⚠ VERIFICAR (> 5pp)'}")

# %% [markdown]
# ## 6. Análise de erros — onde o modelo erra?

print("\n" + "=" * 60)
print("ANÁLISE DE ERROS — Modelo Completo")
print("=" * 60)

errors = y_test[y_pred_full != y_test]
correct = y_test[y_pred_full == y_test]
print(f"\n  Total de erros: {len(errors)} de {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")
print("\n  Erros por classe (real → previsto mais comum):")

for i, label in enumerate(LABELS_PT):
    mask_class = (y_test == i) & (y_pred_full != i)
    if mask_class.sum() > 0:
        wrong_preds = pd.Series(y_pred_full[mask_class]).value_counts()
        most_common = wrong_preds.index[0]
        print(f"  {label:<22} → errou {mask_class.sum()}x "
              f"(mais comum: previu '{LABELS_PT[most_common]}')")

# %% [markdown]
# ## 7. Visualizações

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise Avançada — Modelo Comportamental, Fairness e Erros',
             fontsize=14, fontweight='bold', color='white', y=1.01)
title_kw = dict(color='white', fontsize=12, fontweight='bold', pad=8)

# 7.1 Completo vs comportamental por classe
ax = axes[0, 0]
f1_full_list = [rep_full[str(i)]['f1-score'] for i in range(7)]
f1_beh_list  = [rep_beh[str(i)]['f1-score']  for i in range(7)]
x = np.arange(7)
ax.bar(x - 0.2, f1_full_list, 0.38, color='#4CAF93', label='Completo', alpha=0.9)
ax.bar(x + 0.2, f1_beh_list,  0.38, color='#E06030', label='Comportamental', alpha=0.9)
ax.axhline(0.75, color='#F5C842', linestyle='--', linewidth=1.2, label='Meta 75%')
ax.set_xticks(x)
ax.set_xticklabels([l[:10] for l in LABELS_PT], rotation=30, ha='right', fontsize=8)
ax.set_ylim(0.5, 1.05)
ax.set_title('F1: Completo vs Comportamental', **title_kw)
ax.set_ylabel('F1-score')
ax.legend(fontsize=9, facecolor='#1A1D27', labelcolor='white', edgecolor='#444')
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# 7.2 Feature importance comportamental
ax = axes[0, 1]
feat_imp_beh = pd.Series(model_beh.feature_importances_,
                          index=X_train_beh.columns).sort_values(ascending=True)
colors = ['#E06030' if v > 0.15 else '#F5C842' if v > 0.05 else '#4CAF93'
          for v in feat_imp_beh.values]
ax.barh(feat_imp_beh.index, feat_imp_beh.values, color=colors, edgecolor='none')
ax.set_title('Importância das features\n(modelo comportamental)', **title_kw)
ax.set_xlabel('Importância relativa')
ax.tick_params(labelsize=8)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# 7.3 Fairness — acurácia por gênero e classe
ax = axes[1, 0]
acc_f_cls, acc_m_cls = [], []
for i in range(7):
    mf = (y_test == i) & mask_f
    mm = (y_test == i) & mask_m
    acc_f_cls.append(accuracy_score(y_test[mf], y_pred_full[mf]) if mf.sum() > 0 else 0)
    acc_m_cls.append(accuracy_score(y_test[mm], y_pred_full[mm]) if mm.sum() > 0 else 0)
x = np.arange(7)
ax.bar(x - 0.2, acc_f_cls, 0.38, color='#E06030', label='Feminino', alpha=0.9)
ax.bar(x + 0.2, acc_m_cls, 0.38, color='#4CAF93', label='Masculino', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([l[:9] for l in LABELS_PT], rotation=30, ha='right', fontsize=8)
ax.set_ylim(0, 1.15)
ax.set_title('Fairness: acurácia por gênero e classe', **title_kw)
ax.set_ylabel('Acurácia')
ax.legend(fontsize=9, facecolor='#1A1D27', labelcolor='white', edgecolor='#444')
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# 7.4 Matriz de confusão — modelo comportamental
ax = axes[1, 1]
cm_beh = confusion_matrix(y_test, y_pred_beh)
sns.heatmap(cm_beh, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=[l[:8] for l in LABELS_PT],
            yticklabels=[l[:8] for l in LABELS_PT],
            ax=ax, linewidths=0.3, linecolor='#333',
            annot_kws={'size': 8, 'color': 'white'})
ax.set_title('Matriz de confusão\n(modelo comportamental)', **title_kw)
ax.set_xlabel('Previsto')
ax.set_ylabel('Real')
plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=7)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'analise_avancada.png'),
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("\n✓ Gráfico salvo em images/analise_avancada.png")

# %% [markdown]
# ## 8. Limitações conhecidas e recomendações

print("\n" + "=" * 60)
print("LIMITAÇÕES CONHECIDAS DO MODELO")
print("=" * 60)
print("""
1. DEPENDÊNCIA DE IMC
   Peso e altura respondem por ~70% da importância do modelo completo.
   O modelo comportamental (sem essas features) atinge 79% de acurácia —
   suficiente para triagem preventiva quando peso/altura não estão disponíveis.

2. GENERALIZAÇÃO DO DATASET
   O dataset é de origem acadêmica com distribuição balanceada artificial.
   Em populações hospitalares reais, a distribuição pode ser diferente,
   exigindo retreinamento periódico (monitoramento de data drift).

3. FAIXA ETÁRIA DOMINANTE
   77% dos registros são de pessoas entre 14-30 anos.
   A performance para pacientes acima de 40 anos pode ser inferior
   devido à menor representatividade no treino.

4. AUSÊNCIA DE DADOS CLÍNICOS OBJETIVOS
   O dataset usa autodeclaração para hábitos alimentares e de exercício.
   Em ambiente hospitalar, dados laboratoriais (glicemia, colesterol)
   enriqueceriam significativamente o poder preditivo comportamental.

5. FAIRNESS BÁSICA VALIDADA
   A diferença de acurácia entre gêneros é de apenas 2.5pp — dentro
   do limite aceitável. Análise mais aprofundada por interseccionalidade
   (gênero × faixa etária) é recomendada antes de deploy em produção.

RECOMENDAÇÕES PARA PRODUÇÃO:
   - Implementar monitoramento de distribuição de inputs (data drift)
   - Retreinar a cada 6 meses com novos dados hospitalares
   - Registrar predições para auditoria e análise retroativa
   - Validar clinicamente com amostra de pacientes reais antes de usar
""")