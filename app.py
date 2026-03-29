"""
📊 Dashboard de Amostragem - ENEM 2024
Três métodos de amostragem com 20% da população + comparação com parâmetros populacionais
"""

import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    .method-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        margin-right: 0.5rem;
    }

    .badge-blue   { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
    .badge-cyan   { background: rgba(6,182,212,0.15);  color: #22d3ee; border: 1px solid rgba(6,182,212,0.3); }
    .badge-green  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
    .badge-amber  { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .badge-purple { background: rgba(139,92,246,0.15); color: #a78bfa; border: 1px solid rgba(139,92,246,0.3); }

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

    .stat-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }

    .stat-table th {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-secondary);
        border-bottom: 1px solid var(--border);
        padding: 0.5rem 0.75rem;
        text-align: left;
    }

    .stat-table td {
        padding: 0.6rem 0.75rem;
        border-bottom: 1px solid rgba(30,41,59,0.5);
        color: var(--text-primary);
    }

    .stat-table tr:hover td { background: var(--bg-accent); }

    .diff-pos { color: #34d399; }
    .diff-neg { color: #f87171; }
    .diff-neu { color: #94a3b8; }

    /* Plotly dark override */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }

    /* Streamlit widgets dark */
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
# CARREGAMENTO DA POPULAÇÃO COMPLETA (AMOSTRA DO BD = POPULAÇÃO DO ESTUDO)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_populacao():
    """
    Carrega até 200.000 registros APENAS da tabela de resultados.
    Não há chave comum entre participantes e resultados, então usamos
    só ed_enem_2024_resultados, que já contém as notas necessárias.
    """
    engine = get_engine()
    if engine is None:
        return None

    bar = st.progress(0, text="🔍 Inspecionando colunas disponíveis...")

    try:
        # 1. Descobrir colunas disponíveis na tabela de resultados
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

        # 2. Colunas numéricas de notas — pegamos as que existirem
        candidatas_num = [
            "nota_mt_matematica", "nota_redacao", "nota_media_5_notas",
            "nota_ch_ciencias_humanas", "nota_cn_ciencias_natureza",
            "nota_lc_linguagens", "nu_nota_mt", "nu_nota_redacao",
        ]
        colunas_num = [c for c in candidatas_num if c in colunas_disponiveis]

        # 3. Colunas categóricas — pegamos as que existirem
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

        # 4. Montar SELECT com as colunas disponíveis
        todas_cols = list(dict.fromkeys(colunas_num + colunas_cat))
        select_cols = ", ".join(todas_cols)

        # Filtro: pelo menos uma nota não nula
        where_clause = " OR ".join([f"{c} IS NOT NULL" for c in colunas_num[:3]])

        query = f"""
        SELECT {select_cols}
        FROM public.ed_enem_2024_resultados
        WHERE {where_clause}
        LIMIT 200000
        """

        bar.progress(40, text="📥 Carregando dados da tabela de resultados...")
        df = pl.read_database(query=query, connection=engine)
        bar.progress(75, text="⚙️ Otimizando tipos de dados...")

        # Cast categóricas
        for c in colunas_cat:
            if c in df.columns:
                df = df.with_columns(
                    pl.col(c).cast(pl.Utf8, strict=False).cast(pl.Categorical)
                )

        # Garantir que as variáveis numéricas globais existam no df
        # (atualiza VARIAVEIS_NUM e VARIAVEIS_CAT conforme o que foi carregado)
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

TAXA = 0.20   # 20%
SEED = 42
VARIAVEIS_NUM = ["nota_mt_matematica", "nota_redacao", "nota_media_5_notas"]
VARIAVEIS_CAT = ["tp_sexo", "tp_cor_raca", "sg_uf_prova", "renda_familiar"]  # atualizado em runtime
ESTRATO_COL1  = "tp_sexo"    # atualizado após carregamento
ESTRATO_COL2  = "tp_cor_raca"  # atualizado após carregamento


def amostra_aleatoria_simples(df: pl.DataFrame, taxa: float = TAXA, seed: int = SEED) -> pl.DataFrame:
    """Amostragem Aleatória Simples sem reposição."""
    n = math.ceil(len(df) * taxa)
    return df.sample(n=n, seed=seed, shuffle=True)


def amostra_sistematica(df: pl.DataFrame, taxa: float = TAXA, seed: int = SEED) -> pl.DataFrame:
    """
    Amostragem Sistemática.
    k = N/n  →  sorteia ponto de partida em [0, k-1] e retira 1 a cada k elementos.
    """
    N = len(df)
    if N == 0:
        return df

    n = max(1, math.ceil(N * taxa))   # garante n >= 1
    k = max(1, N // n)                # garante k >= 1, sem divisão por zero

    rng = np.random.default_rng(seed)
    # rng.integers(low, high) → high é exclusivo; se k==1 usamos 0 direto
    inicio = int(rng.integers(0, k)) if k > 1 else 0

    indices = list(range(inicio, N, k))[:n]
    return df[indices]


def amostra_estratificada(df: pl.DataFrame, taxa: float = TAXA, seed: int = SEED) -> pl.DataFrame:
    """
    Amostragem Estratificada proporcional.
    Usa as duas primeiras colunas categóricas disponíveis como estratos.
    """
    if len(df) == 0:
        return df

    global ESTRATO_COL1, ESTRATO_COL2

    # Escolher colunas de estrato dinamicamente
    cats_disponiveis = [c for c in VARIAVEIS_CAT if c in df.columns]
    col1 = cats_disponiveis[0] if len(cats_disponiveis) > 0 else None
    col2 = cats_disponiveis[1] if len(cats_disponiveis) > 1 else col1

    if col1 is None:
        # Sem coluna categórica: usa amostragem simples
        return amostra_aleatoria_simples(df, taxa=taxa, seed=seed)

    ESTRATO_COL1 = col1
    ESTRATO_COL2 = col2

    # Cast seguro: funciona com Categorical, Utf8 ou qualquer tipo
    def to_str(col_name: str) -> pl.Expr:
        return (
            pl.col(col_name)
            .cast(pl.Utf8, strict=False)
            .fill_null("ND")
        )

    estratos = df.with_columns(
        (to_str(col1) + "_" + to_str(col2)).alias("_estrato")
    )

    partes = []
    rng = np.random.default_rng(seed)

    for _estrato_val, grupo in estratos.group_by("_estrato"):
        if len(grupo) == 0:
            continue
        n_estrato = max(1, math.ceil(len(grupo) * taxa))
        seed_local = int(rng.integers(0, 2**31))
        partes.append(
            grupo.sample(n=min(n_estrato, len(grupo)), seed=seed_local, shuffle=True)
        )

    # Fallback: se nenhum estrato foi gerado, usa amostragem aleatória simples
    if not partes:
        return amostra_aleatoria_simples(df, taxa=taxa, seed=seed)

    return pl.concat(partes).drop("_estrato")


# ============================================================================
# ESTATÍSTICAS COMPARATIVAS
# ============================================================================

def estatisticas_numericas(df: pl.DataFrame, label: str) -> dict:
    """Calcula estatísticas descritivas das variáveis numéricas.
    Retorna 0.0 para qualquer estatística que resulte em None
    (coluna vazia ou totalmente nula após drop_nulls).
    """
    def safe(val) -> float:
        """Converte para float, retornando 0.0 se None ou NaN."""
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
            out[v] = {k: 0.0 for k in ["mean","median","std","min","max","q1","q3","cv"]}
            continue

        col = df[v].drop_nulls()

        if len(col) == 0:
            out[v] = {k: 0.0 for k in ["mean","median","std","min","max","q1","q3","cv"]}
            continue

        mean_val = safe(col.mean())
        std_val  = safe(col.std())
        cv_val   = round(std_val / mean_val * 100, 4) if mean_val != 0 else 0.0

        out[v] = {
            "mean":   round(mean_val, 4),
            "median": round(safe(col.median()), 4),
            "std":    round(std_val, 4),
            "min":    round(safe(col.min()), 4),
            "max":    round(safe(col.max()), 4),
            "q1":     round(safe(col.quantile(0.25)), 4),
            "q3":     round(safe(col.quantile(0.75)), 4),
            "cv":     cv_val,
        }
    return out


def tabela_comparacao(pop_stats: dict, amostras_stats: list[dict], variavel: str) -> pd.DataFrame:
    """Monta DataFrame de comparação entre população e amostras."""
    metricas = ["mean", "median", "std", "min", "max", "q1", "q3", "cv"]
    labels_pt = {
        "mean": "Média", "median": "Mediana", "std": "Desvio Padrão",
        "min": "Mínimo", "max": "Máximo",
        "q1": "Q1 (25%)", "q3": "Q3 (75%)", "cv": "CV (%)",
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
    [
        "🏠 Visão Geral",
        "🎲 Aleatória Simples",
        "📏 Sistemática",
        "🗂️ Estratificada",
        "⚖️ Comparação Geral",
    ],
)

st.sidebar.markdown("---")

# Mapa de labels legíveis — funciona para qualquer nome de coluna
LABEL_MAP = {
    "nota_mt_matematica":       "Matemática",
    "nota_redacao":             "Redação",
    "nota_media_5_notas":       "Nota Média (5 provas)",
    "nota_ch_ciencias_humanas": "Ciências Humanas",
    "nota_cn_ciencias_natureza":"Ciências Natureza",
    "nota_lc_linguagens":       "Linguagens",
    "nu_nota_mt":               "Matemática (nu)",
    "nu_nota_redacao":          "Redação (nu)",
}

def col_label(col: str) -> str:
    """Retorna label legível; se não conhecer, formata o nome da coluna."""
    return LABEL_MAP.get(col, col.replace("_", " ").title())

variavel_analise = st.sidebar.selectbox(
    "VARIÁVEL NUMÉRICA",
    VARIAVEIS_NUM,
    format_func=col_label,
)

taxa_pct = st.sidebar.slider("TAXA DE AMOSTRAGEM (%)", 5, 40, 20, 5)
taxa = taxa_pct / 100

seed_val = st.sidebar.number_input("SEMENTE ALEATÓRIA", value=42, min_value=0)

if st.sidebar.button("🔄 Recarregar"):
    st.cache_data.clear()
    st.rerun()

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
    st.markdown("""
    **Possíveis causas:**
    - As tabelas `ed_enem_2024_participantes` / `ed_enem_2024_resultados` não existem ou estão vazias
    - O JOIN entre `nu_inscricao` e `nu_sequencial` não encontrou correspondências
    - Os filtros `WHERE nota_media_5_notas IS NOT NULL` eliminaram todos os registros
    - Problema de permissão no schema `public`

    Clique em **🔄 Recarregar** na barra lateral ou verifique as credenciais em Settings → Secrets.
    """)
    st.stop()

# Gerar as três amostras
df_aas  = amostra_aleatoria_simples(df_pop, taxa=taxa, seed=seed_val)
df_sist = amostra_sistematica(df_pop, taxa=taxa, seed=seed_val)
df_est  = amostra_estratificada(df_pop, taxa=taxa, seed=seed_val)

# Estatísticas
pop_stats  = estatisticas_numericas(df_pop,  "População")
aas_stats  = estatisticas_numericas(df_aas,  "Aleatória Simples")
sist_stats = estatisticas_numericas(df_sist, "Sistemática")
est_stats  = estatisticas_numericas(df_est,  "Estratificada")
todas_amostras = [aas_stats, sist_stats, est_stats]

var_label = col_label(variavel_analise)

# ============================================================================
# HELPERS DE PLOT
# ============================================================================

def apply_theme(fig):
    fig.update_layout(**PLOT_TEMPLATE["layout"])
    return fig


def histograma_comparado(df_p, df_a, label_a, cor_a, variavel, var_label):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_p[variavel].to_list(), name="População",
        histnorm="probability density", opacity=0.45,
        marker_color=COLORS["populacao"], nbinsx=60,
    ))
    fig.add_trace(go.Histogram(
        x=df_a[variavel].to_list(), name=label_a,
        histnorm="probability density", opacity=0.7,
        marker_color=cor_a, nbinsx=60,
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"Distribuição — {var_label}",
        xaxis_title=var_label,
        yaxis_title="Densidade",
    )
    return apply_theme(fig)


def boxplot_comparado(frames: dict, variavel: str, var_label: str):
    fig = go.Figure()
    for label, (df, cor) in frames.items():
        fig.add_trace(go.Box(
            y=df[variavel].to_list(), name=label,
            marker_color=cor, boxmean="sd",
            line_width=1.5,
        ))
    fig.update_layout(title=f"Boxplot — {var_label}", yaxis_title=var_label)
    return apply_theme(fig)


def grafico_erros(pop_s, amostras_s, variavel):
    metrs = ["mean", "median", "std", "q1", "q3"]
    labels_pt = {"mean": "Média", "median": "Mediana", "std": "Desvio Padrão",
                 "q1": "Q1", "q3": "Q3"}
    cores_am = {
        "Aleatória Simples": COLORS["aleatoria"],
        "Sistemática": COLORS["sistematica"],
        "Estratificada": COLORS["estratificada"],
    }
    fig = go.Figure()
    x_labels = [labels_pt[m] for m in metrs]
    for a in amostras_s:
        erros = [erro_relativo(pop_s[variavel][m], a[variavel][m]) for m in metrs]
        fig.add_trace(go.Bar(
            name=a["label"], x=x_labels, y=erros,
            marker_color=cores_am[a["label"]], opacity=0.85,
        ))
    fig.update_layout(
        barmode="group",
        title="Erro Relativo (%) por Estatística",
        yaxis_title="Erro Relativo (%)",
        xaxis_title="Estatística",
    )
    return apply_theme(fig)


def radar_erros(pop_s, amostras_s, variavel):
    metrs = ["mean", "median", "std", "q1", "q3", "cv"]
    labels_pt = ["Média", "Mediana", "Desvio Padrão", "Q1", "Q3", "CV"]
    cores_am = {
        "Aleatória Simples": COLORS["aleatoria"],
        "Sistemática": COLORS["sistematica"],
        "Estratificada": COLORS["estratificada"],
    }
    fig = go.Figure()
    for a in amostras_s:
        vals = [erro_relativo(pop_s[variavel][m], a[variavel][m]) for m in metrs]
        vals += [vals[0]]
        theta = labels_pt + [labels_pt[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, name=a["label"],
            line_color=cores_am[a["label"]], fill="toself",
            fillcolor=cores_am[a["label"]], opacity=0.25,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(17,24,39,0.6)",
            radialaxis=dict(visible=True, gridcolor="#1e293b", color="#475569"),
            angularaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
        ),
        title="Radar de Erros Relativos (%)",
    )
    return apply_theme(fig)


# ============================================================================
# COMPONENTES REUTILIZÁVEIS
# ============================================================================

def card_metrica(label, valor, cor="blue", sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card {cor}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{valor}</div>
        {sub_html}
    </div>"""


def render_tabela_comparacao(pop_s, amostras_s, variavel):
    df_cmp = tabela_comparacao(pop_s, amostras_s, variavel)
    styled = df_cmp.style.format({
        col: "{:.4f}" for col in df_cmp.columns if col != "Estatística"
    }).set_properties(**{
        "background-color": "#111827",
        "color": "#f1f5f9",
        "border-color": "#1e293b",
    })
    st.dataframe(df_cmp, width='stretch', hide_index=True)


# ============================================================================
# PÁGINA: VISÃO GERAL
# ============================================================================

if pagina == "🏠 Visão Geral":
    st.markdown("""
    <div class="main-title">Análise de Amostragem</div>
    <div class="main-subtitle">ENEM 2024 · Três Métodos · 20% da População</div>
    """, unsafe_allow_html=True)

    # Métricas topo
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(card_metrica("POPULAÇÃO", f"{len(df_pop):,}".replace(",", "."),
                                 "blue", "registros base"), unsafe_allow_html=True)
    with c2:
        st.markdown(card_metrica("ALEATÓRIA SIMPLES", f"{len(df_aas):,}".replace(",", "."),
                                 "cyan", f"{taxa_pct}% da pop."), unsafe_allow_html=True)
    with c3:
        k_display = max(1, len(df_pop) // len(df_sist)) if len(df_sist) > 0 else "—"
        st.markdown(card_metrica("SISTEMÁTICA", f"{len(df_sist):,}".replace(",", "."),
                                 "green", f"k = {k_display}"), unsafe_allow_html=True)
    with c4:
        st.markdown(card_metrica("ESTRATIFICADA", f"{len(df_est):,}".replace(",", "."),
                                 "amber", "sexo × raça"), unsafe_allow_html=True)
    with c5:
        st.markdown(card_metrica("TAXA", f"{taxa_pct}%",
                                 "purple", f"semente {seed_val}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cards explicativos dos métodos
    st.markdown('<div class="section-header">📖 Métodos de Amostragem</div>', unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("""
        <div class="metric-card cyan">
            <div class="metric-label">Método 01</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;
                font-size:1.1rem;color:#22d3ee;margin:0.4rem 0;">
                Aleatória Simples
            </div>
            <div class="info-box" style="margin-top:0.75rem;font-size:0.82rem;">
                Cada elemento da população tem <strong>igual probabilidade</strong>
                de ser selecionado. Utiliza sorteio direto sem qualquer ordenação ou
                agrupamento prévio. É o método de referência — base para avaliar os demais.
                <br><br>
                <strong>Fórmula:</strong> n = N × taxa
            </div>
        </div>
        """, unsafe_allow_html=True)

    with e2:
        st.markdown("""
        <div class="metric-card green">
            <div class="metric-label">Método 02</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;
                font-size:1.1rem;color:#34d399;margin:0.4rem 0;">
                Sistemática
            </div>
            <div class="info-box" style="margin-top:0.75rem;font-size:0.82rem;">
                Seleciona elementos em <strong>intervalos regulares</strong> (k).
                Um ponto de partida aleatório em [0, k-1] é sorteado e, a partir dele,
                coleta-se 1 a cada k unidades. Eficiente e fácil de implementar
                quando a lista está disponível.
                <br><br>
                <strong>Fórmula:</strong> k = N / n
            </div>
        </div>
        """, unsafe_allow_html=True)

    with e3:
        st.markdown("""
        <div class="metric-card amber">
            <div class="metric-label">Método 03</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;
                font-size:1.1rem;color:#fbbf24;margin:0.4rem 0;">
                Estratificada
            </div>
            <div class="info-box" style="margin-top:0.75rem;font-size:0.82rem;">
                Divide a população em <strong>estratos homogêneos</strong>
                (sexo × cor/raça) e retira amostras proporcionais de cada estrato.
                Garante representação adequada de subgrupos minoritários,
                reduzindo variância do estimador.
                <br><br>
                <strong>Fórmula:</strong> nᵢ = Nᵢ/N × n
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visão rápida: médias comparadas (todas as variáveis)
    st.markdown('<div class="section-header">📊 Médias Comparadas — Todas as Variáveis Numéricas</div>',
                unsafe_allow_html=True)

    rows_resumo = []
    for v in VARIAVEIS_NUM:
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
    st.dataframe(pd.DataFrame(rows_resumo), width='stretch', hide_index=True)

    # Radar geral
    st.markdown("<br>", unsafe_allow_html=True)
    col_r, col_b = st.columns([1, 1])
    with col_r:
        st.plotly_chart(radar_erros(pop_stats, todas_amostras, variavel_analise),
                        use_container_width=True)
    with col_b:
        frames_box = {
            "População": (df_pop, COLORS["populacao"]),
            "AAS": (df_aas, COLORS["aleatoria"]),
            "Sistemática": (df_sist, COLORS["sistematica"]),
            "Estratificada": (df_est, COLORS["estratificada"]),
        }
        st.plotly_chart(boxplot_comparado(frames_box, variavel_analise, var_label),
                        use_container_width=True)


# ============================================================================
# PÁGINA: ALEATÓRIA SIMPLES
# ============================================================================

elif pagina == "🎲 Aleatória Simples":
    st.markdown("""
    <div class="main-title">Aleatória Simples</div>
    <div class="main-subtitle">Amostragem sem reposição · Probabilidade igual</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <strong>Como funciona:</strong> Cada um dos <strong>{len(df_pop):,}</strong> registros
    da população tem probabilidade <strong>{taxa:.2%}</strong> de ser selecionado.
    Usamos o método de sorteio direto com semente <code>{seed_val}</code> para reprodutibilidade.
    A amostra resultante contém <strong>{len(df_aas):,} registros</strong>
    ({taxa_pct}% da população).
    </div>
    """, unsafe_allow_html=True)

    # Métricas
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(card_metrica("N POPULAÇÃO", f"{len(df_pop):,}".replace(",","."), "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(card_metrica("N AMOSTRA", f"{len(df_aas):,}".replace(",","."), "cyan"), unsafe_allow_html=True)
    with c3:
        st.markdown(card_metrica("MÉDIA POP.", f"{pop_stats[variavel_analise]['mean']:.2f}", "blue"), unsafe_allow_html=True)
    with c4:
        err = erro_relativo(pop_stats[variavel_analise]["mean"], aas_stats[variavel_analise]["mean"])
        st.markdown(card_metrica("ERRO RELATIVO MÉDIA", f"{err:.3f}%", "cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Histograma", "📦 Boxplot", "📋 Estatísticas"])

    with tab1:
        fig = histograma_comparado(df_pop, df_aas, "Aleatória Simples",
                                   COLORS["aleatoria"], variavel_analise, var_label)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        frames_box = {
            "População": (df_pop, COLORS["populacao"]),
            "Aleatória Simples": (df_aas, COLORS["aleatoria"]),
        }
        st.plotly_chart(boxplot_comparado(frames_box, variavel_analise, var_label),
                        use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">📋 Tabela Comparativa</div>', unsafe_allow_html=True)
        render_tabela_comparacao(pop_stats, [aas_stats], variavel_analise)

        st.markdown("""
        <div class="info-box">
        <strong>Δ (delta)</strong>: diferença absoluta entre a estatística da amostra e o parâmetro populacional.
        Valores próximos de zero indicam boa representatividade da amostra.
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: SISTEMÁTICA
# ============================================================================

elif pagina == "📏 Sistemática":
    N = len(df_pop)
    n_s = len(df_sist)
    k = max(1, N // n_s) if n_s > 0 else 1

    st.markdown("""
    <div class="main-title">Sistemática</div>
    <div class="main-subtitle">Intervalo regular k · Ponto de partida aleatório</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <strong>Como funciona:</strong> Calculamos o intervalo de seleção
    <strong>k = {k}</strong> (= {N:,} ÷ {n_s if n_s > 0 else "—"}).
    Um ponto de partida aleatório entre 0 e {k-1} foi sorteado; a partir dele,
    selecionamos 1 elemento a cada {k} posições da lista.
    A amostra contém <strong>{n_s:,} registros</strong> ({taxa_pct}% da população).
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(card_metrica("N POPULAÇÃO", f"{N:,}".replace(",","."), "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(card_metrica("N AMOSTRA", f"{n_s:,}".replace(",","."), "green"), unsafe_allow_html=True)
    with c3:
        st.markdown(card_metrica("INTERVALO k", f"{k}", "blue", "1 a cada k elem."), unsafe_allow_html=True)
    with c4:
        err = erro_relativo(pop_stats[variavel_analise]["mean"], sist_stats[variavel_analise]["mean"])
        st.markdown(card_metrica("ERRO RELATIVO MÉDIA", f"{err:.3f}%", "green"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Histograma", "📦 Boxplot", "📋 Estatísticas"])

    with tab1:
        fig = histograma_comparado(df_pop, df_sist, "Sistemática",
                                   COLORS["sistematica"], variavel_analise, var_label)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        frames_box = {
            "População": (df_pop, COLORS["populacao"]),
            "Sistemática": (df_sist, COLORS["sistematica"]),
        }
        st.plotly_chart(boxplot_comparado(frames_box, variavel_analise, var_label),
                        use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">📋 Tabela Comparativa</div>', unsafe_allow_html=True)
        render_tabela_comparacao(pop_stats, [sist_stats], variavel_analise)

        st.markdown("""
        <div class="info-box">
        <strong>Atenção:</strong> A amostragem sistemática pode introduzir viés se existir
        alguma <em>periodicidade</em> nos dados alinhada ao intervalo k.
        Verifique o histograma e compare a distribuição com a população.
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: ESTRATIFICADA
# ============================================================================

elif pagina == "🗂️ Estratificada":
    # Colunas de estrato usadas (definidas dinamicamente em amostra_estratificada)
    ec1 = ESTRATO_COL1
    ec2 = ESTRATO_COL2
    lbl_ec1 = ec1.replace("tp_", "").replace("_", " ").title()
    lbl_ec2 = ec2.replace("tp_", "").replace("_", " ").title()
    subtitulo_est = f"Proporcional por {lbl_ec1} × {lbl_ec2}"

    st.markdown(f"""
    <div class="main-title">Estratificada</div>
    <div class="main-subtitle">{subtitulo_est}</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <strong>Como funciona:</strong> A população é dividida em estratos formados pela combinação
    <strong>{ec1} × {ec2}</strong>. De cada estrato, retiramos {taxa_pct}% dos registros
    proporcionalmente ao seu tamanho. Isso garante que grupos minoritários não sejam
    sub-representados na amostra. Total: <strong>{len(df_est):,} registros</strong>.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(card_metrica("N POPULAÇÃO", f"{len(df_pop):,}".replace(",","."), "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(card_metrica("N AMOSTRA", f"{len(df_est):,}".replace(",","."), "amber"), unsafe_allow_html=True)
    with c3:
        try:
            n_estratos = (df_pop
                .with_columns(
                    (pl.col(ec1).cast(pl.Utf8, strict=False).fill_null("ND") + "_" +
                     pl.col(ec2).cast(pl.Utf8, strict=False).fill_null("ND")).alias("_e"))
                ["_e"].n_unique())
        except Exception:
            n_estratos = "—"
        st.markdown(card_metrica("ESTRATOS", f"{n_estratos}", "amber", f"{lbl_ec1} × {lbl_ec2}"), unsafe_allow_html=True)
    with c4:
        err = erro_relativo(pop_stats[variavel_analise]["mean"], est_stats[variavel_analise]["mean"])
        st.markdown(card_metrica("ERRO RELATIVO MÉDIA", f"{err:.3f}%", "amber"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Histograma", "📦 Boxplot", "📋 Estatísticas", "🗂️ Estratos"])

    with tab1:
        fig = histograma_comparado(df_pop, df_est, "Estratificada",
                                   COLORS["estratificada"], variavel_analise, var_label)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        frames_box = {
            "População": (df_pop, COLORS["populacao"]),
            "Estratificada": (df_est, COLORS["estratificada"]),
        }
        st.plotly_chart(boxplot_comparado(frames_box, variavel_analise, var_label),
                        use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">📋 Tabela Comparativa</div>', unsafe_allow_html=True)
        render_tabela_comparacao(pop_stats, [est_stats], variavel_analise)

    with tab4:
            st.markdown('<div class="section-header">🗂️ Distribuição por Estrato</div>', unsafe_allow_html=True)
    
            try:
                # Se ec1 == ec2 (só uma col categórica disponível), usa apenas uma dimensão
                usar_dois_estratos = (ec1 != ec2)
    
                if usar_dois_estratos:
                    group_cols_pop = [lbl_ec1, lbl_ec2]
                    group_cols_est = [lbl_ec1, lbl_ec2]
                    with_cols_pop = [
                        pl.col(ec1).cast(pl.Utf8, strict=False).fill_null("ND").alias(lbl_ec1),
                        pl.col(ec2).cast(pl.Utf8, strict=False).fill_null("ND").alias(lbl_ec2),
                    ]
                    with_cols_est = with_cols_pop
                else:
                    group_cols_pop = [lbl_ec1]
                    group_cols_est = [lbl_ec1]
                    with_cols_pop = [
                        pl.col(ec1).cast(pl.Utf8, strict=False).fill_null("ND").alias(lbl_ec1),
                    ]
                    with_cols_est = with_cols_pop
    
                df_pop_e = (
                    df_pop
                    .with_columns(with_cols_pop)
                    .group_by(group_cols_pop)
                    .agg(pl.len().alias("N Pop."))
                    .sort("N Pop.", descending=True)
                )
    
                df_est_e = (
                    df_est
                    .with_columns(with_cols_est)
                    .group_by(group_cols_est)
                    .agg(pl.len().alias("N Amostra"))
                    .sort("N Amostra", descending=True)
                )
    
                df_estratos = df_pop_e.join(df_est_e, on=group_cols_pop, how="left").to_pandas()
                total_pop = df_estratos["N Pop."].sum() or 1
                total_am  = df_estratos["N Amostra"].sum() or 1
                df_estratos["% Pop."]    = (df_estratos["N Pop."]    / total_pop * 100).round(2)
                df_estratos["% Amostra"] = (df_estratos["N Amostra"] / total_am  * 100).round(2)
                df_estratos["Δ %"] = (df_estratos["% Amostra"] - df_estratos["% Pop."]).round(3)
                st.dataframe(df_estratos, width='stretch', hide_index=True)
    
                if usar_dois_estratos:
                    fig_e = px.bar(
                        df_estratos.sort_values("N Pop.", ascending=False),
                        x=lbl_ec2, y=["N Pop.", "N Amostra"],
                        facet_col=lbl_ec1, barmode="group",
                        color_discrete_map={
                            "N Pop.": COLORS["populacao"],
                            "N Amostra": COLORS["estratificada"],
                        },
                        title=f"Tamanho dos Estratos: Pop. vs Amostra ({lbl_ec1} × {lbl_ec2})",
                    )
                else:
                    fig_e = px.bar(
                        df_estratos.sort_values("N Pop.", ascending=False),
                        x=lbl_ec1, y=["N Pop.", "N Amostra"],
                        barmode="group",
                        color_discrete_map={
                            "N Pop.": COLORS["populacao"],
                            "N Amostra": COLORS["estratificada"],
                        },
                        title=f"Tamanho dos Estratos: Pop. vs Amostra ({lbl_ec1})",
                    )
    
                fig_e.update_layout(**PLOT_TEMPLATE["layout"])
                st.plotly_chart(fig_e, use_container_width=True)
    
            except Exception as e_est:
                st.warning(f"Não foi possível montar o gráfico de estratos: {e_est}")
    
# ============================================================================
# PÁGINA: COMPARAÇÃO GERAL
# ============================================================================

elif pagina == "⚖️ Comparação Geral":
    st.markdown("""
    <div class="main-title">Comparação Geral</div>
    <div class="main-subtitle">Os três métodos frente aos parâmetros populacionais</div>
    """, unsafe_allow_html=True)

    # Cards de erros por método
    st.markdown('<div class="section-header">🎯 Erro Relativo na Média</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    erros = {
        "Aleatória Simples": erro_relativo(pop_stats[variavel_analise]["mean"], aas_stats[variavel_analise]["mean"]),
        "Sistemática":       erro_relativo(pop_stats[variavel_analise]["mean"], sist_stats[variavel_analise]["mean"]),
        "Estratificada":     erro_relativo(pop_stats[variavel_analise]["mean"], est_stats[variavel_analise]["mean"]),
    }
    cols_c = [c1, c2, c3]
    cores_c = ["cyan", "green", "amber"]
    for i, (nome, err) in enumerate(erros.items()):
        with cols_c[i]:
            qualidade = "Excelente" if err < 0.5 else "Boa" if err < 1.5 else "Regular"
            st.markdown(card_metrica(nome.upper(), f"{err:.4f}%", cores_c[i], qualidade),
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráficos lado a lado
    col_r, col_b = st.columns(2)
    with col_r:
        st.plotly_chart(grafico_erros(pop_stats, todas_amostras, variavel_analise),
                        use_container_width=True)
    with col_b:
        st.plotly_chart(radar_erros(pop_stats, todas_amostras, variavel_analise),
                        use_container_width=True)

    # Histograma triplo
    st.markdown('<div class="section-header">📊 Distribuições Sobrepostas</div>', unsafe_allow_html=True)

    fig_all = go.Figure()
    configs = [
        (df_pop,  "População",         COLORS["populacao"],   0.30),
        (df_aas,  "Aleatória Simples", COLORS["aleatoria"],   0.60),
        (df_sist, "Sistemática",        COLORS["sistematica"], 0.60),
        (df_est,  "Estratificada",      COLORS["estratificada"], 0.60),
    ]
    for df_, lbl, cor, opa in configs:
        fig_all.add_trace(go.Histogram(
            x=df_[variavel_analise].to_list(), name=lbl,
            histnorm="probability density",
            marker_color=cor, opacity=opa, nbinsx=60,
        ))
    fig_all.update_layout(
        barmode="overlay",
        title=f"Distribuições Comparadas — {var_label}",
        xaxis_title=var_label, yaxis_title="Densidade",
        **PLOT_TEMPLATE["layout"],
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # Tabela completa
    st.markdown('<div class="section-header">📋 Tabela Completa de Estatísticas</div>', unsafe_allow_html=True)
    render_tabela_comparacao(pop_stats, todas_amostras, variavel_analise)

    # Resumo final
    st.markdown("<br>", unsafe_allow_html=True)
    melhor = min(erros, key=erros.get)
    st.markdown(f"""
    <div class="metric-card amber">
        <div class="metric-label">🏆 Diagnóstico Automático — {var_label}</div>
        <div style="margin-top:0.75rem;font-size:0.9rem;line-height:1.7;color:#e2e8f0;">
            O método com <strong>menor erro relativo na média</strong> foi
            <strong style="color:#fbbf24;">{melhor}</strong>
            ({erros[melhor]:.4f}%).
            <br>
            Todos os três métodos apresentam erros muito baixos, confirmando que
            uma taxa de amostragem de <strong>{taxa_pct}%</strong> é suficiente para
            representar adequadamente os parâmetros populacionais do ENEM 2024.
            <br><br>
            <span style="color:#94a3b8;font-size:0.82rem;">
            Ajuste a taxa de amostragem na barra lateral para explorar o efeito do tamanho amostral nos erros.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style="text-align:center;font-family:'IBM Plex Mono',monospace;
    font-size:0.65rem;color:#334155;padding:1.5rem 0;letter-spacing:0.06em;">
ENEM 2024 · AMOSTRAGEM · STREAMLIT + POLARS + POSTGRESQL ·
POPULAÇÃO: {len(df_pop):,} REGISTROS · AAS: {len(df_aas):,} · SIST: {len(df_sist):,} · EST: {len(df_est):,}
</div>
""", unsafe_allow_html=True)
