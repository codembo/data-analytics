# =============================================================================
# TECH CHALLENGE — FASE 4 | FIAP POSTECH DATA ANALYTICS
# App Streamlit v2: Sistema Preditivo + Dashboard Analítico (melhorado)
# =============================================================================

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'data',   'Obesity.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'modelo_obesidade.pkl')
MODEL_BEH   = os.path.join(BASE_DIR, 'models', 'modelo_comportamental.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
META_PATH   = os.path.join(BASE_DIR, 'models', 'metadata.json')

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistente de Diagnóstico Nutricional | FIAP",
    page_icon="🏥", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
  .disclaimer {
    background: #1a2a1a; border: 1px solid #4CAF93; border-radius: 8px;
    padding: 10px 16px; font-size: 0.85rem; color: #b0d4b0; margin-bottom: 12px;
  }
  .result-box {
    border-radius: 14px; padding: 28px; text-align: center; margin: 12px 0;
  }
  .result-normal { background:#1a3a2a; border:2px solid #4CAF93; }
  .result-over   { background:#3a3a1a; border:2px solid #F5C842; }
  .result-obese  { background:#3a1a1a; border:2px solid #E06030; }
  .kpi-delta { font-size:0.8rem; color:#aaa; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ── Cache ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():      return joblib.load(MODEL_PATH)
@st.cache_resource
def load_model_beh():
    if os.path.exists(MODEL_BEH): return joblib.load(MODEL_BEH)
    return None
@st.cache_resource
def load_scaler():     return joblib.load(SCALER_PATH)
@st.cache_data
def load_data():       return pd.read_csv(DATA_PATH)
@st.cache_data
def load_meta():
    with open(META_PATH) as f: return json.load(f)

model      = load_model()
model_beh  = load_model_beh()
scaler     = load_scaler()
df_raw     = load_data()
meta       = load_meta()

ORDER     = meta['order']
LABELS_PT = meta['labels_pt']
PALETTE   = ['#4CAF93','#2E9E6B','#F5C842','#E8973A','#E06030','#C03820','#8B1A0A']

@st.cache_data
def prepare_df():
    df = df_raw.copy()
    for col in ['FCVC','NCP','CH2O','FAF','TUE']:
        df[col] = df[col].round().astype(int)
    df['BMI']       = (df['Weight'] / df['Height']**2).round(1)
    df['Obesity_ord'] = df['Obesity'].map({v:i for i,v in enumerate(ORDER)})
    df['Classe_PT'] = df['Obesity'].map(dict(zip(ORDER, LABELS_PT)))
    df['Genero']    = df['Gender'].map({'Female':'Feminino','Male':'Masculino'})
    df['Faixa_Etaria'] = pd.cut(df['Age'],
                                 bins=[0,20,30,40,61],
                                 labels=['14–20','21–30','31–40','41+'])
    return df

df = prepare_df()

# ── Predição ──────────────────────────────────────────────────────────────────
def predict(inputs: dict, use_behavioral=False):
    row = {col: 0 for col in meta['feature_cols']}
    row['Gender']         = 1 if inputs['gender'] == 'Masculino' else 0
    row['family_history'] = 1 if inputs['family_history'] == 'Sim' else 0
    row['FAVC']           = 1 if inputs['favc'] == 'Sim' else 0
    row['SMOKE']          = 1 if inputs['smoke'] == 'Sim' else 0
    row['SCC']            = 1 if inputs['scc'] == 'Sim' else 0
    row['Age']    = inputs['age']
    row['Height'] = inputs['height']
    row['Weight'] = inputs['weight']
    row['FCVC']   = inputs['fcvc']
    row['NCP']    = inputs['ncp']
    row['CH2O']   = inputs['ch2o']
    row['FAF']    = inputs['faf']
    row['TUE']    = inputs['tue']
    freq_map = {'Não':0,'Às vezes':1,'Frequentemente':2,'Sempre':3}
    row['CAEC'] = freq_map[inputs['caec']]
    row['CALC'] = freq_map[inputs['calc']]
    mtrans_map = {
        'Transporte público':'MTRANS_Public_Transportation',
        'Automóvel':'MTRANS_Automobile','A pé':'MTRANS_Walking',
        'Bicicleta':'MTRANS_Bike','Moto':'MTRANS_Motorbike',
    }
    row[mtrans_map[inputs['mtrans']]] = 1

    X = pd.DataFrame([row])[meta['feature_cols']]
    X_sc = scaler.transform(X)

    if use_behavioral and model_beh:
        X_beh = pd.DataFrame(X_sc, columns=meta['feature_cols'])
        X_beh = X_beh.drop(columns=['Weight','Height'])
        pred  = model_beh.predict(X_beh)[0]
        proba = model_beh.predict_proba(X_beh)[0]
    else:
        pred  = model.predict(X_sc)[0]
        proba = model.predict_proba(X_sc)[0]
    return pred, proba

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
col_title, col_badge = st.columns([3,1])
with col_title:
    st.markdown("## 🏥 Assistente de Diagnóstico Nutricional")
    st.markdown("**FIAP PósTech — Data Analytics | Tech Challenge Fase 4**")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("Modelo GBM · Acurácia 95.7%")

st.markdown("""
<div class="disclaimer">
⚕️ <b>Aviso clínico:</b> Esta ferramenta é um <b>sistema de apoio à decisão clínica</b>
e não substitui a avaliação do profissional de saúde. O diagnóstico final é de
responsabilidade exclusiva do médico ou nutricionista responsável.
</div>
""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔬 Sistema Preditivo",
    "📊 Dashboard Analítico",
    "🧠 Transparência do Modelo",
])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — SISTEMA PREDITIVO
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Diagnóstico por paciente")

    with st.sidebar:
        st.markdown("## 📋 Dados do Paciente")
        st.divider()
        st.markdown("**Informações pessoais**")
        gender = st.selectbox("Gênero", ["Feminino","Masculino"])
        age    = st.slider("Idade (anos)", 14, 65, 25)
        height = st.slider("Altura (m)", 1.45, 2.00, 1.70, step=0.01,
                           format="%.2f")
        weight = st.slider("Peso (kg)", 39, 175, 75)

        st.divider()
        st.markdown("**Hábitos alimentares**")
        family_history = st.selectbox("Histórico familiar de sobrepeso?", ["Sim","Não"])
        favc = st.selectbox("Come alimentos calóricos com frequência?", ["Sim","Não"])
        fcvc = st.selectbox("Vegetais nas refeições (1=raramente · 3=sempre)", [1,2,3], index=1)
        ncp  = st.selectbox("Refeições principais por dia", [1,2,3,4], index=2)
        caec = st.selectbox("Come entre as refeições?",
                            ["Não","Às vezes","Frequentemente","Sempre"], index=1)

        st.divider()
        st.markdown("**Estilo de vida**")
        smoke = st.selectbox("Fuma?", ["Não","Sim"])
        ch2o  = st.selectbox("Água/dia (1 = <1L · 2 = 1-2L · 3 = >2L)", [1,2,3], index=1)
        scc   = st.selectbox("Monitora calorias?", ["Não","Sim"])
        faf   = st.selectbox("Atividade física (0=nenhuma · 3=diária)", [0,1,2,3], index=1)
        tue   = st.selectbox("Telas/dia (0=0-2h · 1=3-5h · 2=>5h)", [0,1,2], index=1)
        calc  = st.selectbox("Consome álcool?",
                             ["Não","Às vezes","Frequentemente","Sempre"], index=1)
        mtrans = st.selectbox("Meio de transporte",
                              ["Transporte público","Automóvel","A pé","Bicicleta","Moto"])

        st.divider()
        use_beh = st.toggle("🔬 Modo comportamental (sem peso/altura)",
                            value=False,
                            help="Usa apenas hábitos de vida para predição. "
                                 "Útil para triagem preventiva. Acurácia: 79%")
        analisar = st.button("🔍 Analisar Paciente",
                             use_container_width=True, type="primary")

    if analisar:
        inputs = dict(
            gender=gender, age=age, height=height, weight=weight,
            family_history=family_history, favc=favc, fcvc=fcvc,
            ncp=ncp, caec=caec, smoke=smoke, ch2o=ch2o, scc=scc,
            faf=faf, tue=tue, calc=calc, mtrans=mtrans,
        )
        pred_idx, proba = predict(inputs, use_behavioral=use_beh)
        pred_pt  = LABELS_PT[pred_idx]
        bmi      = round(weight / height**2, 1)

        if pred_idx <= 1:   box_class, emoji = "result-normal", "✅"
        elif pred_idx <= 3: box_class, emoji = "result-over",   "⚠️"
        else:               box_class, emoji = "result-obese",  "🚨"

        if use_beh:
            st.info("🔬 Resultado baseado apenas em hábitos comportamentais (sem peso/altura) — Acurácia: 79%")

        col_res, col_prob = st.columns([1,1])

        with col_res:
            st.markdown(f"""
            <div class="result-box {box_class}">
                <div style="font-size:2.8rem">{emoji}</div>
                <div style="font-size:1.8rem; font-weight:800; margin:10px 0; color:white">
                    {pred_pt}
                </div>
                <div style="color:#ccc; font-size:0.95rem">Diagnóstico previsto pelo modelo</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("IMC", f"{bmi}", delta=None)
            c2.metric("Confiança", f"{proba[pred_idx]*100:.1f}%")
            c3.metric("Acurácia modelo", "95.7%" if not use_beh else "79.0%")

            st.divider()
            st.markdown("##### 📋 Orientação clínica")
            orientacoes = {
                0: ("🟢 Abaixo do peso",
                    "Avaliação nutricional recomendada para investigar causas. "
                    "Considerar triagem para distúrbios alimentares ou metabólicos."),
                1: ("🟢 Peso saudável",
                    "Manter os hábitos alimentares e de atividade física atuais. "
                    "Consulta de rotina anual recomendada."),
                2: ("🟡 Sobrepeso Grau I",
                    "Orientação nutricional e aumento gradual da atividade física. "
                    "Monitoramento do IMC a cada 3 meses."),
                3: ("🟡 Sobrepeso Grau II",
                    "Intervenção nutricional estruturada recomendada. "
                    "Avaliação de risco cardiovascular indicada."),
                4: ("🔴 Obesidade Grau I",
                    "Acompanhamento médico regular obrigatório. "
                    "Plano de emagrecimento multidisciplinar (nutrição + exercício)."),
                5: ("🔴 Obesidade Grau II",
                    "Risco cardiovascular e metabólico elevado. "
                    "Tratamento multidisciplinar urgente. Avaliar comorbidades."),
                6: ("🔴 Obesidade Grau III (mórbida)",
                    "Risco de vida elevado. Avaliação para cirurgia bariátrica "
                    "pode ser indicada. Acompanhamento intensivo obrigatório."),
            }
            titulo_or, texto_or = orientacoes[pred_idx]
            st.markdown(f"**{titulo_or}**")
            st.info(texto_or)

            # Fatores de risco identificados
            riscos = []
            if family_history == 'Sim': riscos.append("🧬 Histórico familiar positivo")
            if favc == 'Sim':           riscos.append("🍔 Consumo frequente de alimentos calóricos")
            if faf <= 1:                riscos.append("🛋️ Baixa frequência de atividade física")
            if caec in ['Frequentemente','Sempre']: riscos.append("🍿 Lanches frequentes entre refeições")
            if calc in ['Frequentemente','Sempre']: riscos.append("🍺 Consumo frequente de álcool")
            if mtrans == 'Automóvel':   riscos.append("🚗 Transporte sedentário")

            if riscos:
                st.markdown("**⚠️ Fatores de risco identificados:**")
                for r in riscos:
                    st.markdown(f"- {r}")

        with col_prob:
            fig_prob = go.Figure(go.Bar(
                x=proba * 100, y=LABELS_PT, orientation='h',
                marker_color=PALETTE,
                text=[f"{p*100:.1f}%" for p in proba],
                textposition='outside',
            ))
            fig_prob.update_layout(
                title="Probabilidade por classe de obesidade",
                xaxis_title="Probabilidade (%)",
                plot_bgcolor='#1A1D27', paper_bgcolor='#0F1117',
                font_color='white', height=400,
                margin=dict(l=120, r=80, t=40, b=40),
                xaxis=dict(range=[0, 115]),
            )
            # Destacar classe predita
            fig_prob.add_vline(x=proba[pred_idx]*100, line_dash="dash",
                               line_color="white", opacity=0.4)
            st.plotly_chart(fig_prob, use_container_width=True)

    else:
        st.info("👈 Preencha os dados do paciente no painel lateral e clique em **Analisar Paciente**.")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Algoritmo",   "Gradient Boosting")
        c2.metric("Acurácia",    "95.7%")
        c3.metric("F1-macro",    "95.6%")
        c4.metric("Classes",     "7 níveis")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — DASHBOARD ANALÍTICO
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Painel Analítico — Obesidade")
    st.caption("Visão geral do dataset para suporte à tomada de decisão da equipe médica.")

    # ── Filtros ──────────────────────────────────────────────────────────────
    with st.expander("🔽 Filtros", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filtro_genero = st.multiselect(
                "Gênero", ["Feminino","Masculino"],
                default=["Feminino","Masculino"])
        with fc2:
            filtro_idade = st.multiselect(
                "Faixa etária", ["14–20","21–30","31–40","41+"],
                default=["14–20","21–30","31–40","41+"])
        with fc3:
            filtro_classe = st.multiselect(
                "Classe de obesidade", LABELS_PT, default=LABELS_PT)

    # Aplicar filtros
    dff = df.copy()
    if filtro_genero:
        dff = dff[dff['Genero'].isin(filtro_genero)]
    if filtro_idade:
        dff = dff[dff['Faixa_Etaria'].isin(filtro_idade)]
    if filtro_classe:
        dff = dff[dff['Classe_PT'].isin(filtro_classe)]

    if len(dff) == 0:
        st.warning("Nenhum registro com os filtros selecionados.")
        st.stop()

    st.caption(f"📊 Exibindo **{len(dff):,}** de **{len(df):,}** pacientes")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.divider()
    total    = len(dff)
    pct_ob   = (dff['Obesity'].isin(['Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])).mean()*100
    pct_over = (dff['Obesity'].isin(['Overweight_Level_I','Overweight_Level_II'])).mean()*100
    pct_fam  = (dff['family_history']=='yes').mean()*100
    bmi_med  = dff['BMI'].median()
    idade_med = dff['Age'].median()

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Pacientes na seleção", f"{total:,}")
    k2.metric("Com obesidade", f"{pct_ob:.1f}%",
              delta="▲ Brasil: 22% (IBGE)", delta_color="off")
    k3.metric("Com sobrepeso", f"{pct_over:.1f}%")
    k4.metric("Histórico familiar", f"{pct_fam:.1f}%")
    k5.metric("IMC mediano", f"{bmi_med:.1f}")
    st.divider()

    # ── Gráfico 1 — Distribuição das classes (linha cheia) ────────────────────
    counts = dff['Obesity'].value_counts().reindex(ORDER).fillna(0)
    fig1 = go.Figure(go.Bar(
        x=counts.values, y=LABELS_PT, orientation='h',
        marker_color=PALETTE,
        text=[f"{int(v)} ({v/total*100:.1f}%)" for v in counts.values],
        textposition='outside',
    ))
    fig1.update_layout(
        title="<b>Distribuição das classes de obesidade</b>",
        xaxis_title="Quantidade de pacientes",
        plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
        font_color='white', height=380,
        margin=dict(l=140, r=120, t=50, b=40),
        xaxis=dict(range=[0, counts.max()*1.25]),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Gráfico 2 — IMC por classe (linha cheia) ──────────────────────────────
    fig2 = px.box(
        dff, x='Classe_PT', y='BMI',
        color='Classe_PT', color_discrete_sequence=PALETTE,
        category_orders={'Classe_PT': LABELS_PT},
        points='outliers',
        labels={'Classe_PT':'Classe', 'BMI':'IMC (kg/m²)'},
    )
    fig2.update_layout(
        title="<b>Distribuição do IMC por classe de obesidade</b>",
        plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
        font_color='white', showlegend=False, height=400,
        xaxis=dict(tickangle=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Gráficos 3 e 4 — em 2 colunas ────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        faf_mean = dff.groupby('Obesity')['FAF'].mean().reindex(ORDER)
        fig3 = go.Figure(go.Bar(
            x=LABELS_PT, y=faf_mean.values, marker_color=PALETTE,
            text=[f"{v:.2f}" for v in faf_mean.values], textposition='outside',
        ))
        fig3.update_layout(
            title="<b>Atividade física média por classe</b>",
            yaxis_title="FAF médio (0–3)",
            yaxis=dict(range=[0, 1.6]),
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=380, xaxis=dict(tickangle=25),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        hf = dff.groupby(['Obesity','family_history']).size().unstack(fill_value=0).reindex(ORDER)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name='Com histórico', x=LABELS_PT,
                              y=hf.get('yes',[0]*7), marker_color='#E06030'))
        fig4.add_trace(go.Bar(name='Sem histórico', x=LABELS_PT,
                              y=hf.get('no',[0]*7),  marker_color='#4CAF93'))
        fig4.update_layout(
            barmode='group',
            title="<b>Histórico familiar por classe</b>",
            yaxis_title="Registros",
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=380, xaxis=dict(tickangle=25),
            legend=dict(bgcolor='#1A1D27'),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Gráficos 5 e 6 ───────────────────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        favc_pct = (dff.groupby('Obesity')['FAVC']
                       .apply(lambda x: (x=='yes').mean()*100)
                       .reindex(ORDER))
        fig5 = go.Figure(go.Bar(
            x=LABELS_PT, y=favc_pct.values, marker_color=PALETTE,
            text=[f"{v:.0f}%" for v in favc_pct.values], textposition='outside',
        ))
        fig5.update_layout(
            title="<b>% que consome alimentos calóricos (FAVC)</b>",
            yaxis_title="% de pacientes",
            yaxis=dict(range=[0,115]),
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=380, xaxis=dict(tickangle=25),
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.scatter(
            dff.sample(min(600,len(dff)), random_state=42),
            x='Height', y='Weight', color='Classe_PT',
            color_discrete_sequence=PALETTE,
            category_orders={'Classe_PT': LABELS_PT},
            hover_data=['Age','BMI','Genero'],
            labels={'Height':'Altura (m)','Weight':'Peso (kg)','Classe_PT':'Classe'},
            opacity=0.65,
        )
        fig6.update_layout(
            title="<b>Peso × Altura por classe</b>",
            plot_bgcolor='#1A1D27', paper_bgcolor='#1A1D27',
            font_color='white', height=380,
            legend=dict(bgcolor='#1A1D27', font_size=10),
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 💡 Principais insights para a equipe médica")

    insights = [
        ("🧬", "Histórico familiar é determinante",
         "Pacientes com Obesidade Tipo III têm histórico familiar positivo em mais de 90% dos casos. Recomenda-se triagem preventiva em familiares de primeiro grau."),
        ("🏃", "Sedentarismo acompanha a progressão",
         "A frequência de atividade física (FAF) cai de 1.25 em pessoas com peso normal para 0.64 em Obesidade Tipo III. Intervenções de exercício são a ação mais impactante."),
        ("🍔", "Alimentação calórica cresce com a obesidade",
         "O consumo de alimentos calóricos (FAVC) e lanches entre refeições (CAEC) aumenta progressivamente. Educação nutricional é prioritária nos grupos de risco."),
        ("⚖️", "46% dos pacientes têm obesidade",
         "Proporção muito acima da média nacional de 22% (IBGE 2023). O perfil do dataset representa uma população de risco que demanda atenção preventiva ampliada."),
        ("🚗", "Transporte passivo está associado à obesidade",
         "Uso de automóvel é mais prevalente em pacientes com sobrepeso e obesidade. Políticas de mobilidade ativa podem ter impacto positivo na saúde coletiva."),
        ("💧", "Hidratação adequada está associada a menor obesidade",
         "Consumo de água acima de 2L/dia e monitoramento calórico são comportamentos mais comuns em pessoas com peso normal ou insuficiente."),
    ]

    i1, i2 = st.columns(2)
    for idx, (emoji, titulo, texto) in enumerate(insights):
        col = i1 if idx % 2 == 0 else i2
        with col:
            with st.container(border=True):
                st.markdown(f"**{emoji} {titulo}**")
                st.caption(texto)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — TRANSPARÊNCIA DO MODELO
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Transparência e Limitações do Modelo")
    st.caption("Esta seção documenta o funcionamento, métricas e limitações conhecidas do sistema.")

    # Métricas gerais
    st.markdown("#### Desempenho do modelo")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Algoritmo",         "Gradient Boosting")
    c2.metric("Acurácia (teste)",  "95.7%", delta="+20.7pp vs meta (75%)")
    c3.metric("F1-macro",          "95.6%")
    c4.metric("Validação cruzada", "96.3% ± 0.3%")

    st.divider()

    # Comparativo completo vs comportamental
    st.markdown("#### Modelo completo vs modelo comportamental")
    st.markdown("""
    Uma análise crítica revelou que **peso e altura respondem por ~70% da importância do modelo completo**.
    Isso faz sentido: a classe de obesidade é clinicamente definida pelo IMC (peso/altura²).

    Para validar o valor real dos dados comportamentais, treinamos um segundo modelo **sem peso e altura**:
    """)

    mc1, mc2 = st.columns(2)
    with mc1:
        with st.container(border=True):
            st.markdown("**Modelo Completo** (20 features)")
            st.metric("Acurácia", "95.7%")
            st.metric("F1-macro", "95.6%")
            st.caption("Inclui peso e altura. Uso: quando exame físico já foi feito.")
    with mc2:
        with st.container(border=True):
            st.markdown("**Modelo Comportamental** (18 features)")
            st.metric("Acurácia", "79.0%")
            st.metric("F1-macro", "78.7%")
            st.caption("Apenas hábitos de vida. Uso: triagem preventiva e rastreamento poblacional.")

    st.info("✅ O modelo comportamental supera a meta de 75% da FIAP — o que valida que "
            "os hábitos de vida têm poder preditivo real, independente do peso e altura.")

    st.divider()

    # Fairness
    st.markdown("#### Análise de justiça (Fairness)")
    f1, f2 = st.columns(2)
    with f1:
        with st.container(border=True):
            st.markdown("**Feminino** (n=201 no teste)")
            st.metric("Acurácia", "97.5%")
    with f2:
        with st.container(border=True):
            st.markdown("**Masculino** (n=222 no teste)")
            st.metric("Acurácia", "95.0%")
    st.success("✅ Diferença de 2.5pp entre gêneros — dentro do limite aceitável (<5pp).")

    st.divider()

    # Limitações
    st.markdown("#### Limitações conhecidas")

    limitacoes = [
        ("⚠️", "Dependência de IMC",
         "O modelo completo depende fortemente de peso e altura. Em contextos onde esses dados já estão disponíveis, o cálculo direto do IMC pode ser suficiente. O valor clínico do modelo está nos fatores comportamentais."),
        ("⚠️", "Dataset acadêmico balanceado artificialmente",
         "A distribuição igualitária das 7 classes é artificial. Em populações hospitalares reais, as classes têm distribuições muito diferentes, o que pode afetar a performance em produção."),
        ("⚠️", "Concentração em jovens adultos",
         "77% dos registros são de pessoas entre 14-30 anos. A performance para pacientes acima de 40 anos pode ser inferior devido à menor representatividade no treinamento."),
        ("⚠️", "Dados autodeclarados",
         "Hábitos alimentares e de exercício são baseados em respostas subjetivas. Em ambiente hospitalar, dados clínicos objetivos (exames laboratoriais) enriqueceriam o modelo."),
        ("ℹ️", "Monitoramento em produção não implementado",
         "Para uso real, recomenda-se implementar monitoramento de data drift, logging de predições para auditoria, e retreinamento periódico a cada 6 meses."),
    ]

    for emoji, titulo, texto in limitacoes:
        with st.container(border=True):
            st.markdown(f"**{emoji} {titulo}**")
            st.caption(texto)

    st.divider()
    st.markdown("#### Sobre os dados de treinamento")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total de registros",  "2.111")
    c2.metric("Features utilizadas", "20")
    c3.metric("Divisão treino/teste","80% / 20%")
    c4.metric("Validação",           "5-fold estratificado")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.82rem'>"
    "FIAP PósTech — Data Analytics | Tech Challenge Fase 4 | "
    "Misael Oliveira · Gustavo Bacelar Horita · Álvaro de Freitas Pinto · Victor Fernando Gil<br>"
    "Modelo: GradientBoostingClassifier · Acurácia: 95.7% · "
    "⚕️ Ferramenta de apoio à decisão clínica — não substitui avaliação médica."
    "</p>",
    unsafe_allow_html=True,
)