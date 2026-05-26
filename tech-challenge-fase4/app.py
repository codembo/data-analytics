# =============================================================================
# TECH CHALLENGE — FASE 4 | FIAP POSTECH DATA ANALYTICS
# App Streamlit: Sistema Preditivo + Dashboard Analítico
# =============================================================================

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# ── Caminhos ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data',   'Obesity.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_obesidade.pkl')
SCALER_PATH= os.path.join(BASE_DIR, 'models', 'scaler.pkl')
META_PATH  = os.path.join(BASE_DIR, 'models', 'metadata.json')

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Obesity Predictor | FIAP Tech Challenge",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0F1117; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background-color: #1A1D27; border-radius: 8px 8px 0 0;
    padding: 8px 20px; color: #ccc; font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background-color: #E06030 !important; color: white !important;
  }
  .metric-card {
    background: #1A1D27; border-radius: 12px; padding: 20px;
    border-left: 4px solid #E06030; margin-bottom: 12px;
  }
  .result-box {
    border-radius: 12px; padding: 24px; text-align: center;
    margin: 16px 0; font-size: 1.1rem;
  }
  .result-normal  { background: #1a3a2a; border: 2px solid #4CAF93; }
  .result-over    { background: #3a3a1a; border: 2px solid #F5C842; }
  .result-obese   { background: #3a1a1a; border: 2px solid #E06030; }
  .insight-card {
    background: #1A1D27; border-radius: 10px; padding: 16px;
    margin: 8px 0; border-left: 3px solid #4CAF93;
    color: #E0E0E0 !important;
  }
  .insight-card b {
    color: #FFFFFF !important;
    font-size: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Carregamento de recursos (cached) ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)

model  = load_model()
scaler = load_scaler()
df_raw = load_data()
meta   = load_meta()

ORDER     = meta['order']
LABELS_PT = meta['labels_pt']
PALETTE   = ['#4CAF93','#2E9E6B','#F5C842','#E8973A','#E06030','#C03820','#8B1A0A']
PAL_DICT  = dict(zip(ORDER, PALETTE))

# ── Pré-processamento do df para o dashboard ──────────────────────────────────
@st.cache_data
def prepare_df():
    df = df_raw.copy()
    for col in ['FCVC','NCP','CH2O','FAF','TUE']:
        df[col] = df[col].round().astype(int)
    df['BMI'] = (df['Weight'] / df['Height']**2).round(1)
    df['Obesity_ord'] = df['Obesity'].map({v: i for i, v in enumerate(ORDER)})
    df['Classe_PT'] = df['Obesity'].map(dict(zip(ORDER, LABELS_PT)))
    return df

df = prepare_df()

# ── Função de predição ────────────────────────────────────────────────────────
def predict(inputs: dict):
    """Recebe um dict com os valores do formulário e retorna (classe, proba_array)."""
    row = {col: 0 for col in meta['feature_cols']}

    # Binárias
    row['Gender']         = 1 if inputs['gender'] == 'Masculino' else 0
    row['family_history'] = 1 if inputs['family_history'] == 'Sim' else 0
    row['FAVC']           = 1 if inputs['favc'] == 'Sim' else 0
    row['SMOKE']          = 1 if inputs['smoke'] == 'Sim' else 0
    row['SCC']            = 1 if inputs['scc'] == 'Sim' else 0

    # Numéricas
    row['Age']    = inputs['age']
    row['Height'] = inputs['height']
    row['Weight'] = inputs['weight']
    row['FCVC']   = inputs['fcvc']
    row['NCP']    = inputs['ncp']
    row['CH2O']   = inputs['ch2o']
    row['FAF']    = inputs['faf']
    row['TUE']    = inputs['tue']

    # Ordinais
    freq_map = {'Não': 0, 'Às vezes': 1, 'Frequentemente': 2, 'Sempre': 3}
    row['CAEC'] = freq_map[inputs['caec']]
    row['CALC'] = freq_map[inputs['calc']]

    # One-Hot MTRANS
    mtrans_map = {
        'Transporte público': 'MTRANS_Public_Transportation',
        'Automóvel':          'MTRANS_Automobile',
        'A pé':               'MTRANS_Walking',
        'Bicicleta':          'MTRANS_Bike',
        'Moto':               'MTRANS_Motorbike',
    }
    col_mtrans = mtrans_map[inputs['mtrans']]
    row[col_mtrans] = 1

    X = pd.DataFrame([row])[meta['feature_cols']]
    X_sc = scaler.transform(X)
    pred  = model.predict(X_sc)[0]
    proba = model.predict_proba(X_sc)[0]
    return pred, proba

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏥 Sistema de Predição de Obesidade")
st.markdown("**FIAP PósTech — Data Analytics | Tech Challenge Fase 4**")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🔬 Sistema Preditivo", "📊 Dashboard Analítico"])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — SISTEMA PREDITIVO
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Diagnóstico de Nível de Obesidade")
    st.markdown("Preencha os dados do paciente no painel lateral e clique em **Analisar**.")

    # ── Sidebar — formulário ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📋 Dados do Paciente")
        st.divider()

        st.markdown("**Informações pessoais**")
        gender = st.selectbox("Gênero", ["Feminino", "Masculino"])
        age    = st.slider("Idade (anos)", 14, 65, 25)
        height = st.slider("Altura (m)", 1.45, 2.00, 1.70, step=0.01)
        weight = st.slider("Peso (kg)", 39, 175, 75)

        st.divider()
        st.markdown("**Histórico e hábitos alimentares**")
        family_history = st.selectbox("Histórico familiar de sobrepeso?", ["Sim","Não"])
        favc = st.selectbox("Come alimentos calóricos com frequência?", ["Sim","Não"])
        fcvc = st.selectbox("Frequência de vegetais nas refeições (1=raramente, 3=sempre)",
                            [1, 2, 3], index=1)
        ncp  = st.selectbox("Refeições principais por dia", [1, 2, 3, 4], index=2)
        caec = st.selectbox("Come entre as refeições?",
                            ["Não","Às vezes","Frequentemente","Sempre"], index=1)

        st.divider()
        st.markdown("**Saúde e estilo de vida**")
        smoke = st.selectbox("Fuma?", ["Não","Sim"])
        ch2o  = st.selectbox("Consumo diário de água (1=<1L, 2=1-2L, 3=>2L)",
                             [1, 2, 3], index=1)
        scc   = st.selectbox("Monitora calorias ingeridas?", ["Não","Sim"])
        faf   = st.selectbox("Frequência de atividade física (0=nenhuma, 3=diária)",
                             [0, 1, 2, 3], index=1)
        tue   = st.selectbox("Horas/dia em telas (0=0-2h, 1=3-5h, 2=>5h)",
                             [0, 1, 2], index=1)
        calc  = st.selectbox("Consome álcool?",
                             ["Não","Às vezes","Frequentemente","Sempre"], index=1)
        mtrans = st.selectbox("Meio de transporte habitual",
                              ["Transporte público","Automóvel","A pé","Bicicleta","Moto"])

        st.divider()
        analisar = st.button("🔍 Analisar Paciente", use_container_width=True, type="primary")

    # ── Área de resultado ─────────────────────────────────────────────────────
    if analisar:
        inputs = dict(
            gender=gender, age=age, height=height, weight=weight,
            family_history=family_history, favc=favc, fcvc=fcvc,
            ncp=ncp, caec=caec, smoke=smoke, ch2o=ch2o, scc=scc,
            faf=faf, tue=tue, calc=calc, mtrans=mtrans,
        )
        pred_idx, proba = predict(inputs)
        pred_en  = ORDER[pred_idx]
        pred_pt  = LABELS_PT[pred_idx]
        bmi      = round(weight / height**2, 1)

        # Cor do resultado
        if pred_idx <= 1:
            box_class = "result-normal"
            emoji = "✅"
        elif pred_idx <= 3:
            box_class = "result-over"
            emoji = "⚠️"
        else:
            box_class = "result-obese"
            emoji = "🚨"

        col_res, col_prob = st.columns([1, 1])

        with col_res:
            st.markdown(f"""
            <div class="result-box {box_class}">
                <div style="font-size:2.5rem">{emoji}</div>
                <div style="font-size:1.6rem; font-weight:700; margin:8px 0">{pred_pt}</div>
                <div style="opacity:0.75">Diagnóstico previsto pelo modelo</div>
            </div>
            """, unsafe_allow_html=True)

            # Métricas do paciente
            c1, c2, c3 = st.columns(3)
            c1.metric("IMC calculado", f"{bmi}")
            c2.metric("Confiança", f"{proba[pred_idx]*100:.1f}%")
            c3.metric("Acurácia do modelo", "95.7%")

            # Interpretação clínica
            st.divider()
            st.markdown("#### 📋 Interpretação clínica")
            interpretacoes = {
                0: "Paciente abaixo do peso. Avaliação nutricional recomendada para investigar causas.",
                1: "Peso dentro da faixa saudável. Manter hábitos alimentares e de atividade física.",
                2: "Sobrepeso leve. Recomenda-se orientação nutricional e aumento da atividade física.",
                3: "Sobrepeso moderado. Intervenção nutricional e monitoramento do IMC são indicados.",
                4: "Obesidade Grau I. Acompanhamento médico regular e plano de emagrecimento estruturado.",
                5: "Obesidade Grau II. Risco cardiovascular elevado. Tratamento multidisciplinar indicado.",
                6: "Obesidade Grau III (mórbida). Avaliação para tratamento cirúrgico pode ser necessária.",
            }
            st.info(interpretacoes[pred_idx])

        with col_prob:
            # Gráfico de probabilidades
            fig_prob = go.Figure(go.Bar(
                x=proba * 100,
                y=LABELS_PT,
                orientation='h',
                marker_color=PALETTE,
                text=[f"{p*100:.1f}%" for p in proba],
                textposition='outside',
            ))
            fig_prob.update_layout(
                title="Probabilidade por classe (%)",
                xaxis_title="Probabilidade (%)",
                plot_bgcolor='#1A1D27',
                paper_bgcolor='#0F1117',
                font_color='white',
                height=380,
                margin=dict(l=120, r=60, t=40, b=40),
                xaxis=dict(range=[0, 110]),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

    else:
        st.info("👈 Preencha os dados do paciente no painel lateral e clique em **Analisar Paciente**.")

        # Métricas do modelo
        st.markdown("#### Sobre o modelo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algoritmo",      "Gradient Boosting")
        c2.metric("Acurácia",       "95.7%")
        c3.metric("F1-macro",       "95.6%")
        c4.metric("Classes",        "7 níveis")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — DASHBOARD ANALÍTICO
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Painel Analítico — Obesidade")
    st.markdown("Visão geral do dataset para suporte à tomada de decisão da equipe médica.")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total    = len(df)
    pct_ob   = (df['Obesity'].isin(['Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])).mean()*100
    pct_over = (df['Obesity'].isin(['Overweight_Level_I','Overweight_Level_II'])).mean()*100
    pct_fam  = (df['family_history'] == 'yes').mean()*100
    bmi_med  = df['BMI'].median()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total de pacientes", f"{total:,}")
    k2.metric("Com obesidade",      f"{pct_ob:.1f}%")
    k3.metric("Com sobrepeso",      f"{pct_over:.1f}%")
    k4.metric("Histórico familiar", f"{pct_fam:.1f}%")
    k5.metric("IMC mediano",        f"{bmi_med}")

    st.divider()

    # ── Linha 1: distribuição + gênero ───────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        counts = df['Obesity'].value_counts().reindex(ORDER)
        fig1 = go.Figure(go.Bar(
            x=counts.values,
            y=LABELS_PT,
            orientation='h',
            marker_color=PALETTE,
            text=counts.values,
            textposition='outside',
        ))
        fig1.update_layout(
            title="Distribuição das classes de obesidade",
            xaxis_title="Quantidade de pacientes",
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=340,
            margin=dict(l=130, r=60, t=40, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        g_cnt = df['Gender'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=['Feminino','Masculino'],
            values=g_cnt.values,
            marker_colors=['#E06030','#4CAF93'],
            hole=0.4,
        ))
        fig2.update_layout(
            title="Distribuição por gênero",
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=340,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Linha 2: IMC + atividade física ──────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(
            df, x='Classe_PT', y='BMI',
            color='Classe_PT',
            color_discrete_sequence=PALETTE,
            category_orders={'Classe_PT': LABELS_PT},
            title="Distribuição do IMC por classe",
            labels={'Classe_PT':'Classe','BMI':'IMC (kg/m²)'},
        )
        fig3.update_layout(
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', showlegend=False, height=360,
            xaxis=dict(tickangle=30),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        faf_mean = df.groupby('Obesity')['FAF'].mean().reindex(ORDER)
        fig4 = go.Figure(go.Bar(
            x=LABELS_PT, y=faf_mean.values,
            marker_color=PALETTE,
            text=[f"{v:.2f}" for v in faf_mean.values],
            textposition='outside',
        ))
        fig4.update_layout(
            title="Frequência média de atividade física por classe",
            yaxis_title="FAF médio (0–3)",
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=360,
            xaxis=dict(tickangle=30),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Linha 3: histórico familiar + transporte ──────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        hf = df.groupby(['Obesity','family_history']).size().unstack(fill_value=0).reindex(ORDER)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name='Com histórico', x=LABELS_PT,
                              y=hf.get('yes', [0]*7), marker_color='#E06030'))
        fig5.add_trace(go.Bar(name='Sem histórico', x=LABELS_PT,
                              y=hf.get('no',  [0]*7), marker_color='#4CAF93'))
        fig5.update_layout(
            barmode='group', title="Histórico familiar por classe",
            yaxis_title="Registros",
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=360,
            xaxis=dict(tickangle=30),
            legend=dict(bgcolor='#1A1D27'),
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        trans_pct = (df.groupby(['Obesity','MTRANS']).size()
                       .unstack(fill_value=0).reindex(ORDER)
                       .div(df.groupby('Obesity').size().reindex(ORDER), axis=0) * 100)
        fig6 = go.Figure()
        colors_t = ['#E06030','#8B1A0A','#4CAF93','#F5C842','#2E9E6B']
        for col_t, color in zip(trans_pct.columns, colors_t):
            label_map = {
                'Automobile':'Automóvel','Walking':'A pé',
                'Public_Transportation':'Transp. público',
                'Bike':'Bicicleta','Motorbike':'Moto'
            }
            fig6.add_trace(go.Bar(
                name=label_map.get(col_t, col_t),
                x=LABELS_PT, y=trans_pct[col_t],
                marker_color=color,
            ))
        fig6.update_layout(
            barmode='stack', title="Meio de transporte por classe (%)",
            yaxis_title="% de pacientes",
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=360,
            xaxis=dict(tickangle=30),
            legend=dict(bgcolor='#1A1D27', font_size=10),
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 💡 Principais insights para a equipe médica")

    insights = [
        ("🧬", "Histórico familiar é determinante",
         "Pacientes com Obesidade Tipo III têm histórico familiar positivo em mais de 90% dos casos. Recomenda-se triagem preventiva em familiares de pacientes obesos."),
        ("🏃", "Sedentarismo acompanha a progressão",
         "A frequência de atividade física (FAF) cai consistentemente à medida que o nível de obesidade aumenta. Intervenções de exercício são a ação mais impactante."),
        ("🍔", "Alimentação calórica cresce com a obesidade",
         "O consumo de alimentos altamente calóricos (FAVC) e lanches entre refeições (CAEC) aumenta progressivamente dos grupos mais leves para os mais graves."),
        ("⚖️", "Dataset bem balanceado",
         "As 7 classes têm distribuição próxima (12-17% cada), o que garante que o modelo aprenda igualmente bem todos os níveis de obesidade sem viés de classe."),
        ("🚗", "Transporte passivo associado à obesidade",
         "O uso de automóvel é mais prevalente em pacientes com sobrepeso e obesidade, enquanto caminhar é mais comum em pessoas com peso normal."),
        ("💧", "Hidratação e monitoramento calórico são protetores",
         "Maior consumo de água (CH2O) e monitoramento de calorias (SCC) estão associados a menores níveis de obesidade no dataset."),
    ]

    i1, i2 = st.columns(2)
    for idx, (emoji, titulo, texto) in enumerate(insights):
        col = i1 if idx % 2 == 0 else i2
        with col:
            with st.container(border=True):
                st.markdown(f"**{emoji} {titulo}**")
                st.caption(texto)

    # ── Scatter interativo peso x altura ─────────────────────────────────────
    st.divider()
    st.markdown("#### Distribuição Peso × Altura (amostra interativa)")
    fig_scatter = px.scatter(
        df.sample(500, random_state=42),
        x='Height', y='Weight',
        color='Classe_PT',
        color_discrete_sequence=PALETTE,
        category_orders={'Classe_PT': LABELS_PT},
        hover_data=['Age','BMI','Gender'],
        labels={'Height':'Altura (m)','Weight':'Peso (kg)','Classe_PT':'Classe'},
        title="Peso × Altura por classe (amostra de 500 pacientes)",
        opacity=0.7,
    )
    fig_scatter.update_layout(
        plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
        font_color='white', height=420,
        legend=dict(bgcolor='#1A1D27'),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#666; font-size:0.85rem'>"
    "FIAP PósTech — Data Analytics | Tech Challenge Fase 4 | "
    "Modelo: GradientBoostingClassifier | Acurácia: 95.7%"
    "</p>",
    unsafe_allow_html=True
)