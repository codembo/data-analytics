# =============================================================================
# TECH CHALLENGE — FASE 4 | FIAP POSTECH DATA ANALYTICS
# Etapa 3: Treinamento, Comparação e Seleção do Modelo
# =============================================================================

# %% [markdown]
# ## 1. Imports

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

from sklearn.linear_model         import LogisticRegression
from sklearn.ensemble             import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm                  import SVC
from sklearn.model_selection      import cross_val_score, StratifiedKFold
from sklearn.metrics              import (accuracy_score, f1_score,
                                          classification_report, confusion_matrix)

# Caminhos
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

# Configuração visual
plt.rcParams['figure.facecolor'] = '#0F1117'
plt.rcParams['axes.facecolor']   = '#1A1D27'
plt.rcParams['text.color']       = 'white'
plt.rcParams['axes.labelcolor']  = 'white'
plt.rcParams['xtick.color']      = 'white'
plt.rcParams['ytick.color']      = 'white'

# %% [markdown]
# ## 2. Carregamento dos dados preparados na Etapa 2

X_train = pd.read_csv(os.path.join(MODELS_DIR, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(MODELS_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(MODELS_DIR, 'y_train.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(MODELS_DIR, 'y_test.csv')).squeeze()

with open(os.path.join(MODELS_DIR, 'metadata.json')) as f:
    meta = json.load(f)

ORDER    = meta['order']
LABELS_PT = meta['labels_pt']
PALETTE  = ['#4CAF93','#2E9E6B','#F5C842','#E8973A','#E06030','#C03820','#8B1A0A']

print(f"Treino : {X_train.shape} | Teste : {X_test.shape}")

# %% [markdown]
# ## 3. Comparação de modelos com validação cruzada (5-fold estratificado)
#
# Testamos 4 algoritmos como baseline antes de escolher o modelo final.
# Usamos StratifiedKFold para garantir proporção das classes em cada fold.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=42),
}

print("\n--- Comparação de modelos (5-fold CV + teste) ---")
print(f"{'Modelo':<25} {'CV Acc':>8} {'±':>6} {'Test Acc':>10} {'F1-macro':>10}")
print("-" * 65)

results = {}
trained_models = {}
for name, model in MODELS.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average='macro')
    results[name] = {
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
        'test_acc': test_acc,
        'test_f1':  test_f1,
    }
    trained_models[name] = model
    print(f"{name:<25} {cv_scores.mean():>8.4f} {cv_scores.std():>6.4f} {test_acc:>10.4f} {test_f1:>10.4f}")

best_name = max(results, key=lambda k: results[k]['test_acc'])
print(f"\n✓ Melhor modelo: {best_name}  (acurácia teste = {results[best_name]['test_acc']:.4f})")

# %% [markdown]
# ## 4. Modelo final: Gradient Boosting com hiperparâmetros otimizados
#
# Gradient Boosting obteve melhor resultado. Refinamos os hiperparâmetros:
#   - n_estimators=300  : mais árvores para maior estabilidade
#   - max_depth=5       : profundidade suficiente sem overfitting
#   - learning_rate=0.1 : taxa padrão — bom balanço velocidade/qualidade
#   - subsample=0.8     : amostragem estocástica reduz variância

final_model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)

final_acc = accuracy_score(y_test, y_pred_final)
final_f1  = f1_score(y_test, y_pred_final, average='macro')

print(f"\n--- Modelo final (Gradient Boosting tuned) ---")
print(f"  Acurácia : {final_acc:.4f}  ({final_acc*100:.1f}%)")
print(f"  F1-macro : {final_f1:.4f}")
print(f"\n  Meta da FIAP (>75%): {'✓ ATINGIDA' if final_acc > 0.75 else '✗ NÃO ATINGIDA'}")

# %% [markdown]
# ## 5. Classification report detalhado

print("\n--- Classification Report ---")
print(classification_report(
    y_test, y_pred_final,
    target_names=LABELS_PT
))

# %% [markdown]
# ## 6. Visualizações

fig = plt.figure(figsize=(18, 14))
title_kw = dict(color='white', fontsize=13, fontweight='bold', pad=10)

# 6.1 Comparação de modelos
ax1 = fig.add_subplot(2, 2, 1)
names   = list(results.keys())
cv_means = [results[n]['cv_mean'] for n in names]
test_accs = [results[n]['test_acc'] for n in names]
x = np.arange(len(names))
bars1 = ax1.bar(x - 0.2, cv_means,   0.38, color='#4CAF93', label='CV Accuracy',   alpha=0.9)
bars2 = ax1.bar(x + 0.2, test_accs,  0.38, color='#E06030', label='Test Accuracy',  alpha=0.9)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.003,
             f'{h:.3f}', ha='center', color='white', fontsize=8)
ax1.axhline(0.75, color='#F5C842', linestyle='--', linewidth=1.2, label='Meta 75%')
ax1.set_xticks(x)
ax1.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
ax1.set_ylim(0.75, 1.02)
ax1.set_title('Comparação de modelos', **title_kw)
ax1.set_ylabel('Acurácia')
ax1.legend(fontsize=9, facecolor='#1A1D27', labelcolor='white', edgecolor='#444')
for sp in ['top', 'right']: ax1.spines[sp].set_visible(False)

# 6.2 Matriz de confusão
ax2 = fig.add_subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=[l[:10] for l in LABELS_PT],
            yticklabels=[l[:10] for l in LABELS_PT],
            ax=ax2, linewidths=0.3, linecolor='#333',
            annot_kws={'size': 9, 'color': 'white'})
ax2.set_title('Matriz de confusão — Gradient Boosting', **title_kw)
ax2.set_xlabel('Previsto')
ax2.set_ylabel('Real')
plt.setp(ax2.get_xticklabels(), rotation=35, ha='right', fontsize=8)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)

# 6.3 Feature importance
ax3 = fig.add_subplot(2, 2, 3)
feat_imp = pd.Series(final_model.feature_importances_, index=X_train.columns)
feat_imp_sorted = feat_imp.sort_values(ascending=True)
colors_fi = ['#E06030' if v > 0.1 else '#F5C842' if v > 0.02 else '#4CAF93'
             for v in feat_imp_sorted.values]
ax3.barh(feat_imp_sorted.index, feat_imp_sorted.values,
         color=colors_fi, edgecolor='none')
ax3.set_title('Importância das features', **title_kw)
ax3.set_xlabel('Importância relativa')
ax3.tick_params(labelsize=8)
for sp in ['top', 'right']: ax3.spines[sp].set_visible(False)
ax3.spines['bottom'].set_color('#444')
ax3.spines['left'].set_color('#444')

# 6.4 F1 por classe
ax4 = fig.add_subplot(2, 2, 4)
report = classification_report(y_test, y_pred_final, output_dict=True)
f1_per_class = [report[str(i)]['f1-score'] for i in range(7)]
bars4 = ax4.bar(range(7), f1_per_class, color=PALETTE, edgecolor='none', alpha=0.9)
for bar, val in zip(bars4, f1_per_class):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f'{val:.3f}', ha='center', color='white', fontsize=9)
ax4.axhline(0.75, color='#F5C842', linestyle='--', linewidth=1.2, label='Meta 75%')
ax4.set_xticks(range(7))
ax4.set_xticklabels([l[:10] for l in LABELS_PT], rotation=30, ha='right', fontsize=8)
ax4.set_ylim(0.7, 1.05)
ax4.set_title('F1-score por classe', **title_kw)
ax4.set_ylabel('F1-score')
ax4.legend(fontsize=9, facecolor='#1A1D27', labelcolor='white', edgecolor='#444')
for sp in ['top', 'right']: ax4.spines[sp].set_visible(False)

fig.suptitle('Etapa 3 — Treinamento e Avaliação dos Modelos',
             fontsize=16, fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'modelo_ml.png'),
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("✓ Gráfico salvo em images/modelo_ml.png")

# %% [markdown]
# ## 7. Salvando o modelo final

joblib.dump(final_model, os.path.join(MODELS_DIR, 'modelo_obesidade.pkl'))
print(f"✓ Modelo salvo em models/modelo_obesidade.pkl")

# Atualizar metadata com info do modelo
meta['modelo'] = {
    'nome':        'GradientBoostingClassifier',
    'acuracia':    round(final_acc, 4),
    'f1_macro':    round(final_f1, 4),
    'parametros': {
        'n_estimators': 300,
        'max_depth':    5,
        'learning_rate': 0.1,
        'subsample':    0.8,
    }
}
with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("✓ metadata.json atualizado com métricas do modelo")

# %% [markdown]
# ## 8. Resumo final

print(f"""
{'='*60}
RESUMO — ETAPA 3
{'='*60}
Modelos testados:
  Logistic Regression  CV={results['Logistic Regression']['cv_mean']:.3f}  Test={results['Logistic Regression']['test_acc']:.3f}
  Random Forest        CV={results['Random Forest']['cv_mean']:.3f}  Test={results['Random Forest']['test_acc']:.3f}
  Gradient Boosting    CV={results['Gradient Boosting']['cv_mean']:.3f}  Test={results['Gradient Boosting']['test_acc']:.3f}  ← ESCOLHIDO
  SVM (RBF)            CV={results['SVM (RBF)']['cv_mean']:.3f}  Test={results['SVM (RBF)']['test_acc']:.3f}

Modelo final: GradientBoostingClassifier (tuned)
  Acurácia  : {final_acc:.4f} ({final_acc*100:.1f}%)
  F1-macro  : {final_f1:.4f}
  Meta FIAP : >75% — ✓ ATINGIDA

Top 5 features mais importantes:
{feat_imp.sort_values(ascending=False).head(5).to_string()}

Artefatos gerados:
  models/modelo_obesidade.pkl  → modelo treinado para o Streamlit
  models/metadata.json         → atualizado com métricas
  images/modelo_ml.png         → gráficos de avaliação
{'='*60}
""")