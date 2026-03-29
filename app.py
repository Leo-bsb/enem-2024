"""
📊 Dashboard de Amostragem - ENEM 2024
Três métodos de amostragem com 20% da população + comparação com parâmetros populacionais
"""

import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import math

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Amostragem ENEM 2024",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS CUSTOMIZADO — Tema escuro, geométrico, editorial
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --bg-primary: #0a0e1a;
        --bg-card: #111827;
        --bg-accent: #1a2235;
        --color-blue: #3b82f6;
        --color-cyan: #06b6d4;
        --color-green: #10b981;
        --color-amber: #f59e0b;
        --color-red: #ef4444;
        --color-purple: #8b5cf6;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: #1e293b;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
    }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }

    .main-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--color-cyan);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }

    .metric-card.blue::before  { background: var(--color-blue); }
    .metric-card.cyan::before  { background: var(--color-cyan); }
    .metric-card.green::before { background: var(--color-green); }
    .metric-card.amber::before { background: var(--color-amber); }
    .metric-card.purple::before{ background: var(--color-purple); }

    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }

    .metric-sub {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.3rem;
    }

    .section-header {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: var(--text-primary);
        padding: 0.75rem 0;
        margin: 1.5rem 0 1rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .info-box {
        background: var(--bg-accent);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        line-height: 1.6;
        color: var(--text-secondary);
    }

    .info-box strong { color: var(--text-primary); }

    .js-plotly-plot .plotly .bg { fill: transparent !important; }

    .stSelectbox label, .stRadio label, .stSlider label {
        color: var(--text-secondary) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    div[data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 8px;
        gap: 4px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        border-radius: 6px;
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-accent) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stDataFrame"] {
        background: var(--bg-card);
        border-radius: 10px;
    }

    .stProgress > div > div > div > div {
        background: var(--color-blue);
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# BANCO DE DADOS
# ============================================================================

PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.6)",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8"),
        title_font=dict(family="Syne, sans-serif", color="#f1f5f9", size=16),
        xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", tickcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", tickcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

COLORS = {
    "populacao": "#3b82f6",
    "aleatoria": "#06b6d4",
    "sistematica": "#10b981",
    "estratificada": "#f59e0b",
}

@st.cache_resource(show_spinner=False)
def get_engine():
    try:
        if hasattr(st, "secrets") and "DB_HOST" in st.secrets:
            cfg = {
                "host": st.secrets["DB_HOST"],
                "database": st.secrets["DB_NAME"],
                "user": st.secrets["DB_USER"],
                "password": st.secrets["DB_PASSWORD"],
                "port": st.secrets.get("DB_PORT", "5432"),
            }
        else:
            cfg = {
                "host": os.getenv("DB_HOST", "bigdata.dataiesb.com"),
                "database": os.getenv("DB_NAME", "iesb"),
                "user": os.getenv("DB_USER", "data_iesb"),
                "password": os.getenv("DB_PASSWORD", "iesb"),
                "port": os.getenv("DB_PORT", "5432"),
            }
        conn_str = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 10, "options": "-c statement_timeout=60000"},
        )
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Erro ao conectar: {e}")
        return None

# ============================================================================
# CARREGAMENTO DA POPULAÇÃO
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_populacao():
    engine = get_engine()
    if engine is None:
        return None

    bar = st.progress(0, text="🔍 Inspecionando colunas disponíveis...")

    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'ed_enem_2024_resultados'
                ORDER BY ordinal_position
            """))
            colunas_disponiveis = [row[0] for row in result]

        bar.progress(20, text=f"✅ {len(colunas_disponiveis)} colunas encontradas")

        candidatas_num = [
            "nota_mt_matematica", "nota_redacao", "nota_media_5_notas",
            "nota_ch_ciencias_humanas", "nota_cn_ciencias_natureza",
            "nota_lc_linguagens", "nu_nota_mt", "nu_nota_redacao",
        ]
        colunas_num = [c for c in candidatas_num if c in colunas_disponiveis]

        candidatas_cat = [
            "tp_sexo", "tp_faixa_etaria", "tp_cor_raca",
            "sg_uf_prova", "tp_presenca_mt", "tp_presenca_lc",
            "tp_status_redacao",
        ]
        colunas_cat = [c for c in candidatas_cat if c in colunas_disponiveis]

        if not colunas_num:
            st.error("❌ Nenhuma coluna de nota reconhecida na tabela de resultados.")
            st.markdown(f"**Colunas disponíveis:** {', '.join(colunas_disponiveis[:30])}")
            bar.empty()
            return None

        todas_cols = list(dict.fromkeys(colunas_num + colunas_cat))
        select_cols = ", ".join(todas_cols)

        where_clause = " OR ".join([f"{c} IS NOT NULL" for c in colunas_num[:3]]) if len(colunas_num) >= 3 \
            else " OR ".join([f"{c} IS NOT NULL" for c in colunas_num])

        query = f"""
        SELECT {select_cols}
        FROM public.ed_enem_2024_resultados
        WHERE {where_clause}
        LIMIT 200000
        """

        bar.progress(40, text="📥 Carregando dados da tabela de resultados...")
        df = pl.read_database(query=query, connection=engine)
        bar.progress(75, text="⚙️ Otimizando tipos de dados...")

        for c in colunas_cat:
            if c in df.columns:
                df = df.with_columns(pl.col(c).cast(pl.Utf8, strict=False).cast(pl.Categorical))

        global VARIAVEIS_NUM, VARIAVEIS_CAT
        VARIAVEIS_NUM = [c for c in colunas_num if c in df.columns]
        VARIAVEIS_CAT = [c for c in colunas_cat if c in df.columns]

        bar.progress(100, text=f"✅ {len(df):,} registros · {len(df.columns)} colunas")
        bar.empty()
        return df

    except Exception as e:
        bar.empty()
        st.error(f"❌ Erro ao carregar dados: {e}")
        return None

# ============================================================================
# MÉTODOS DE AMOSTRAGEM
# ============================================================================

TAXA = 0.20
SEED = 42
VARIAVEIS_NUM = ["nota_mt_matematica", "nota_redacao", "nota_media_5_notas"]
VARIAVEIS_CAT = ["tp_sexo", "tp_cor_raca", "sg_uf_prova", "renda_familiar"]
ESTRATO_COL1 = "tp_sexo"
ESTRATO_COL2 = "tp_cor_raca"

def amostra_aleatoria_simples(df: pl.DataFrame, taxa: float = TAXA, seed: int = SEED) -> pl.DataFrame:
    n = math.ceil(len(df) * taxa)
    return df.sample(n=n, seed=seed, shuffle=True)

def amostra_sistematica(df: pl.DataFrame, taxa: float = TAXA, seed: int = SEED) -> pl.DataFrame:
    N = len(df)
    if N == 0:
        return df
    n = max(1, math.ceil(N * taxa))
    k = max(1, N // n)
    rng = np.random.default_rng(seed)
    inicio = int(rng.integers(0, k)) if k > 1 else 0
    indices = list(range(inicio, N, k))[:n]
    return df[indices]

def amostra_estratificada(df: pl.DataFrame, taxa: float = TAXA, seed: int = SEED) -> pl.DataFrame:
    if len(df) == 0:
        return df

    global ESTRATO_COL1, ESTRATO_COL2
    cats_disponiveis = [c for c in VARIAVEIS_CAT if c in df.columns]
    col1 = cats_disponiveis[0] if len(cats_disponiveis) > 0 else None
    col2 = cats_disponiveis[1] if len(cats_disponiveis) > 1 else col1

    if col1 is None:
        return amostra_aleatoria_simples(df, taxa=taxa, seed=seed)

    ESTRATO_COL1 = col1
    ESTRATO_COL2 = col2

    def to_str(col_name: str) -> pl.Expr:
        return pl.col(col_name).cast(pl.Utf8, strict=False).fill_null("ND")

    estratos = df.with_columns((to_str(col1) + "_" + to_str(col2)).alias("_estrato"))

    partes = []
    rng = np.random.default_rng(seed)

    for _, grupo in estratos.group_by("_estrato"):
        if len(grupo) == 0:
            continue
        n_estrato = max(1, math.ceil(len(grupo) * taxa))
        seed_local = int(rng.integers(0, 2**31))
        partes.append(grupo.sample(n=min(n_estrato, len(grupo)), seed=seed_local, shuffle=True))

    if not partes:
        return amostra_aleatoria_simples(df, taxa=taxa, seed=seed)

    return pl.concat(partes).drop("_estrato")

# ============================================================================
# ESTATÍSTICAS
# ============================================================================

def estatisticas_numericas(df: pl.DataFrame, label: str) -> dict:
    def safe(val) -> float:
        if val is None:
            return 0.0
        try:
            f = float(val)
            return 0.0 if math.isnan(f) else f
        except (TypeError, ValueError):
            return 0.0

    out = {"label": label, "n": len(df)}
    for v in VARIAVEIS_NUM:
        if v not in df.columns:
            out[v] = {k: 0.0 for k in ["mean", "median", "std", "min", "max", "q1", "q3", "cv"]}
            continue

        col = df[v].drop_nulls()
        if len(col) == 0:
            out[v] = {k: 0.0 for k in ["mean", "median", "std", "min", "max", "q1", "q3", "cv"]}
            continue

        mean_val = safe(col.mean())
        std_val = safe(col.std())
        cv_val = round(std_val / mean_val * 100, 4) if mean_val != 0 else 0.0

        out[v] = {
            "mean": round(mean_val, 4),
            "median": round(safe(col.median()), 4),
            "std": round(std_val, 4),
            "min": round(safe(col.min()), 4),
            "max": round(safe(col.max()), 4),
            "q1": round(safe(col.quantile(0.25)), 4),
            "q3": round(safe(col.quantile(0.75)), 4),
            "cv": cv_val,
        }
    return out

def tabela_comparacao(pop_stats: dict, amostras_stats: list[dict], variavel: str) -> pd.DataFrame:
    metricas = ["mean", "median", "std", "min", "max", "q1", "q3", "cv"]
    labels_pt = {
        "mean": "Média", "median": "Mediana", "std": "Desvio Padrão",
        "min": "Mínimo", "max": "Máximo", "q1": "Q1 (25%)", "q3": "Q3 (75%)", "cv": "CV (%)",
    }
    rows = []
    for m in metricas:
        row = {"Estatística": labels_pt[m], "População": pop_stats[variavel][m]}
        for a in amostras_stats:
            val = a[variavel][m]
            diff = val - pop_stats[variavel][m]
            row[a["label"]] = val
            row[f"Δ {a['label']}"] = round(diff, 4)
        rows.append(row)
    return pd.DataFrame(rows)

def erro_relativo(pop_val: float, amst_val: float) -> float:
    if pop_val == 0:
        return 0.0
    return abs((amst_val - pop_val) / pop_val) * 100

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("""
<div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;
    color:#f1f5f9;margin-bottom:0.25rem;">🎲 ENEM 2024</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
    color:#06b6d4;letter-spacing:0.12em;text-transform:uppercase;
    margin-bottom:1rem;">Análise de Amostragem</div>
""", unsafe_allow_html=True)

with st.sidebar:
    with st.spinner("Testando conexão..."):
        engine = get_engine()
    if engine:
        st.success("✅ Banco conectado")
    else:
        st.error("❌ Sem conexão")
        st.stop()

st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "NAVEGAÇÃO",
    ["🏠 Visão Geral", "🎲 Aleatória Simples", "📏 Sistemática", "🗂️ Estratificada", "⚖️ Comparação Geral"],
)

st.sidebar.markdown("---")

LABEL_MAP = {
    "nota_mt_matematica": "Matemática",
    "nota_redacao": "Redação",
    "nota_media_5_notas": "Nota Média (5 provas)",
    "nota_ch_ciencias_humanas": "Ciências Humanas",
    "nota_cn_ciencias_natureza": "Ciências Natureza",
    "nota_lc_linguagens": "Linguagens",
    "nu_nota_mt": "Matemática (nu)",
    "nu_nota_redacao": "Redação (nu)",
}

def col_label(col: str) -> str:
    return LABEL_MAP.get(col, col.replace("_", " ").title())

# IMPORTANTE: selectbox só após carregar df/atualizar variáveis (para evitar inconsistência)
# (mantido mais abaixo)

taxa_pct = st.sidebar.slider("TAXA DE AMOSTRAGEM (%)", 5, 40, 20, 5)
taxa = taxa_pct / 100
seed_val = st.sidebar.number_input("SEMENTE ALEATÓRIA", value=42, min_value=0)

if st.sidebar.button("🔄 Recarregar"):
    st.cache_data.clear()
    st.rerun()

# ============================================================================
# CARREGAR DADOS
# ============================================================================

with st.spinner("Carregando dados..."):
    df_pop = carregar_populacao()

if df_pop is None:
    st.error("Não foi possível carregar os dados.")
    st.stop()

if len(df_pop) == 0:
    st.error("⚠️ O banco retornou 0 registros.")
    st.stop()

if not VARIAVEIS_NUM:
    st.error("❌ Nenhuma variável numérica disponível para análise.")
    st.stop()

variavel_analise = st.sidebar.selectbox("VARIÁVEL NUMÉRICA", VARIAVEIS_NUM, format_func=col_label)
var_label = col_label(variavel_analise)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
    color:#475569;line-height:1.8;">
FONTE: PostgreSQL<br>
bigdata.dataiesb.com<br>
ed_enem_2024_*<br>
CACHE: 1h
</div>
""", unsafe_allow_html=True)

df_aas = amostra_aleatoria_simples(df_pop, taxa=taxa, seed=seed_val)
df_sist = amostra_sistematica(df_pop, taxa=taxa, seed=seed_val)
df_est = amostra_estratificada(df_pop, taxa=taxa, seed=seed_val)

pop_stats = estatisticas_numericas(df_pop, "População")
aas_stats = estatisticas_numericas(df_aas, "Aleatória Simples")
sist_stats = estatisticas_numericas(df_sist, "Sistemática")
est_stats = estatisticas_numericas(df_est, "Estratificada")
todas_amostras = [aas_stats, sist_stats, est_stats]

def apply_theme(fig):
    fig.update_layout(**PLOT_TEMPLATE["layout"])
    return fig

def histograma_comparado(df_p, df_a, label_a, cor_a, variavel, var_label):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_p[variavel].to_list(), name="População", histnorm="probability density", opacity=0.45, marker_color=COLORS["populacao"], nbinsx=60))
    fig.add_trace(go.Histogram(x=df_a[variavel].to_list(), name=label_a, histnorm="probability density", opacity=0.7, marker_color=cor_a, nbinsx=60))
    fig.update_layout(barmode="overlay", title=f"Distribuição — {var_label}", xaxis_title=var_label, yaxis_title="Densidade")
    return apply_theme(fig)

def boxplot_comparado(frames: dict, variavel: str, var_label: str):
    fig = go.Figure()
    for label, (df, cor) in frames.items():
        fig.add_trace(go.Box(y=df[variavel].to_list(), name=label, marker_color=cor, boxmean="sd", line_width=1.5))
    fig.update_layout(title=f"Boxplot — {var_label}", yaxis_title=var_label)
    return apply_theme(fig)

def grafico_erros(pop_s, amostras_s, variavel):
    metrs = ["mean", "median", "std", "q1", "q3"]
    labels_pt = {"mean": "Média", "median": "Mediana", "std": "Desvio Padrão", "q1": "Q1", "q3": "Q3"}
    cores_am = {"Aleatória Simples": COLORS["aleatoria"], "Sistemática": COLORS["sistematica"], "Estratificada": COLORS["estratificada"]}
    fig = go.Figure()
    x_labels = [labels_pt[m] for m in metrs]
    for a in amostras_s:
        erros = [erro_relativo(pop_s[variavel][m], a[variavel][m]) for m in metrs]
        fig.add_trace(go.Bar(name=a["label"], x=x_labels, y=erros, marker_color=cores_am[a["label"]], opacity=0.85))
    fig.update_layout(barmode="group", title="Erro Relativo (%) por Estatística", yaxis_title="Erro Relativo (%)", xaxis_title="Estatística")
    return apply_theme(fig)

def radar_erros(pop_s, amostras_s, variavel):
    metrs = ["mean", "median", "std", "q1", "q3", "cv"]
    labels_pt = ["Média", "Mediana", "Desvio Padrão", "Q1", "Q3", "CV"]
    cores_am = {"Aleatória Simples": COLORS["aleatoria"], "Sistemática": COLORS["sistematica"], "Estratificada": COLORS["estratificada"]}
    fig = go.Figure()
    for a in amostras_s:
        vals = [erro_relativo(pop_s[variavel][m], a[variavel][m]) for m in metrs]
        vals += [vals[0]]
        theta = labels_pt + [labels_pt[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=theta, name=a["label"], line_color=cores_am[a["label"]], fill="toself", fillcolor=cores_am[a["label"]], opacity=0.25))
    fig.update_layout(
        polar=dict(bgcolor="rgba(17,24,39,0.6)", radialaxis=dict(visible=True, gridcolor="#1e293b", color="#475569"), angularaxis=dict(gridcolor="#1e293b", color="#94a3b8")),
        title="Radar de Erros Relativos (%)",
    )
    return apply_theme(fig)

def card_metrica(label, valor, cor="blue", sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""<div class="metric-card {cor}"><div class="metric-label">{label}</div><div class="metric-value">{valor}</div>{sub_html}</div>"""

def render_tabela_comparacao(pop_s, amostras_s, variavel):
    df_cmp = tabela_comparacao(pop_s, amostras_s, variavel)
    st.dataframe(df_cmp, width="stretch", hide_index=True)

if pagina == "🏠 Visão Geral":
    st.markdown("""
    <div class="main-title">Análise de Amostragem</div>
    <div class="main-subtitle">ENEM 2024 · Três Métodos · 20% da População</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(card_metrica("POPULAÇÃO", f"{len(df_pop):,}".replace(",", "."), "blue", "registros base"), unsafe_allow_html=True)
    with c2:
        st.markdown(card_metrica("ALEATÓRIA SIMPLES", f"{len(df_aas):,}".replace(",", "."), "cyan", f"{taxa_pct}% da pop."), unsafe_allow_html=True)
    with c3:
        k_display = max(1, len(df_pop) // len(df_sist)) if len(df_sist) > 0 else "—"
        st.markdown(card_metrica("SISTEMÁTICA", f"{len(df_sist):,}".replace(",", "."), "green", f"k = {k_display}"), unsafe_allow_html=True)
    with c4:
        st.markdown(card_metrica("ESTRATIFICADA", f"{len(df_est):,}".replace(",", "."), "amber", "estratos dinâmicos"), unsafe_allow_html=True)
    with c5:
        st.markdown(card_metrica("TAXA", f"{taxa_pct}%", "purple", f"semente {seed_val}"), unsafe_allow_html=True)

    rows_resumo = []
    for v in VARIAVEIS_NUM:
        if v not in pop_stats:
            continue
        rows_resumo.append({
            "Variável": col_label(v),
            "População": pop_stats[v]["mean"],
            "Aleatória Simples": aas_stats[v]["mean"],
            "Sistemática": sist_stats[v]["mean"],
            "Estratificada": est_stats[v]["mean"],
            "Δ AAS (%)": round(erro_relativo(pop_stats[v]["mean"], aas_stats[v]["mean"]), 3),
            "Δ Sist (%)": round(erro_relativo(pop_stats[v]["mean"], sist_stats[v]["mean"]), 3),
            "Δ Est (%)": round(erro_relativo(pop_stats[v]["mean"], est_stats[v]["mean"]), 3),
        })
    st.dataframe(pd.DataFrame(rows_resumo), width="stretch", hide_index=True)

    col_r, col_b = st.columns([1, 1])
    with col_r:
        st.plotly_chart(radar_erros(pop_stats, todas_amostras, variavel_analise), width="stretch")
    with col_b:
        frames_box = {
            "População": (df_pop, COLORS["populacao"]),
            "AAS": (df_aas, COLORS["aleatoria"]),
            "Sistemática": (df_sist, COLORS["sistematica"]),
            "Estratificada": (df_est, COLORS["estratificada"]),
        }
        st.plotly_chart(boxplot_comparado(frames_box, variavel_analise, var_label), width="stretch")

elif pagina == "🎲 Aleatória Simples":
    st.markdown("""<div class="main-title">Aleatória Simples</div>""", unsafe_allow_html=True)
    fig = histograma_comparado(df_pop, df_aas, "Aleatória Simples", COLORS["aleatoria"], variavel_analise, var_label)
    st.plotly_chart(fig, width="stretch")
    render_tabela_comparacao(pop_stats, [aas_stats], variavel_analise)

elif pagina == "📏 Sistemática":
    N = len(df_pop)
    n_s = len(df_sist)
    k = max(1, N // n_s) if n_s > 0 else 1
    st.markdown("""<div class="main-title">Sistemática</div>""", unsafe_allow_html=True)
    fig = histograma_comparado(df_pop, df_sist, "Sistemática", COLORS["sistematica"], variavel_analise, var_label)
    st.plotly_chart(fig, width="stretch")
    render_tabela_comparacao(pop_stats, [sist_stats], variavel_analise)

elif pagina == "🗂️ Estratificada":
    st.markdown("""<div class="main-title">Estratificada</div>""", unsafe_allow_html=True)
    fig = histograma_comparado(df_pop, df_est, "Estratificada", COLORS["estratificada"], variavel_analise, var_label)
    st.plotly_chart(fig, width="stretch")
    render_tabela_comparacao(pop_stats, [est_stats], variavel_analise)

elif pagina == "⚖️ Comparação Geral":
    st.markdown("""<div class="main-title">Comparação Geral</div>""", unsafe_allow_html=True)
    col_r, col_b = st.columns(2)
    with col_r:
        st.plotly_chart(grafico_erros(pop_stats, todas_amostras, variavel_analise), width="stretch")
    with col_b:
        st.plotly_chart(radar_erros(pop_stats, todas_amostras, variavel_analise), width="stretch")
    render_tabela_comparacao(pop_stats, todas_amostras, variavel_analise)

st.markdown("---")
st.markdown(f"""
<div style="text-align:center;font-family:'IBM Plex Mono',monospace;
    font-size:0.65rem;color:#334155;padding:1.5rem 0;letter-spacing:0.06em;">
ENEM 2024 · AMOSTRAGEM · STREAMLIT + POLARS + POSTGRESQL ·
POPULAÇÃO: {len(df_pop):,} · AAS: {len(df_aas):,} · SIST: {len(df_sist):,} · EST: {len(df_est):,}
</div>
""", unsafe_allow_html=True)
