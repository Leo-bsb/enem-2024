"""
📊 Dashboard de Análise Exploratória - ENEM 2024
Aplicativo Streamlit com análise completa dos dados do ENEM
"""

import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Análise ENEM 2024",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS CUSTOMIZADO
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================================================

@st.cache_data
def carregar_dados():
    """Carrega e faz join dos dados do ENEM"""
    try:
        # Carregar dados
        df_participantes = pl.scan_parquet("ed_enem_2024_participantes.parquet")
        df_resultados = pl.scan_parquet("ed_enem_2024_resultados.parquet")
        
        # Join
        df_completo = df_participantes.join(
            df_resultados,
            left_on="nu_inscricao",
            right_on="nu_sequencial",
            how="inner"
        ).collect()
        
        return df_completo
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_data
def calcular_estatisticas_descritivas(df):
    """Calcula estatísticas descritivas para variáveis numéricas"""
    variaveis = ['nota_mt_matematica', 'nota_redacao', 'nota_media_5_notas', 'idade_calculada']
    
    stats = {}
    for var in variaveis:
        stats[var] = {
            'count': df[var].count(),
            'mean': df[var].mean(),
            'std': df[var].std(),
            'min': df[var].min(),
            'q1': df[var].quantile(0.25),
            'median': df[var].median(),
            'q3': df[var].quantile(0.75),
            'max': df[var].max(),
            'cv': (df[var].std() / df[var].mean() * 100) if df[var].mean() != 0 else 0
        }
    
    return stats

@st.cache_data
def calcular_tabelas_frequencia(df):
    """Calcula tabelas de frequência para variáveis categóricas"""
    variaveis = ['tp_sexo', 'tp_faixa_etaria', 'tp_cor_raca', 'sg_uf_prova', 'renda_familiar']
    
    tabelas = {}
    for var in variaveis:
        freq = (df
            .group_by(var)
            .agg(pl.len().alias('frequencia'))
            .with_columns([
                (pl.col('frequencia') / pl.col('frequencia').sum() * 100).alias('percentual'),
                (pl.col('frequencia').cum_sum() / pl.col('frequencia').sum() * 100).alias('perc_acumulado')
            ])
            .sort('frequencia', descending=True))
        
        tabelas[var] = freq
    
    return tabelas

# ============================================================================
# SIDEBAR - NAVEGAÇÃO
# ============================================================================

st.sidebar.markdown("# 📊 ENEM 2024")
st.sidebar.markdown("### Análise Exploratória de Dados")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegação",
    [
        "🏠 Visão Geral",
        "📊 Distribuições de Frequência",
        "📈 Variáveis Qualitativas",
        "📉 Variáveis Quantitativas",
        "📦 Box Plots",
        "🔗 Análise de Correlação",
        "🎯 Análises Cruzadas",
        "📋 Relatório Completo"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Sobre os Dados:**
- **Inscritos:** 4.332.944
- **Com Resultados:** 2.990.093
- **Taxa de Comparecimento:** 69,01%
""")

# ============================================================================
# CARREGAR DADOS
# ============================================================================

df_completo = carregar_dados()

if df_completo is None:
    st.error("⚠️ Não foi possível carregar os dados. Verifique se os arquivos Parquet estão no diretório correto.")
    st.stop()

# Calcular estatísticas uma vez
stats = calcular_estatisticas_descritivas(df_completo)
tabelas_freq = calcular_tabelas_frequencia(df_completo)

# ============================================================================
# PÁGINA 1: VISÃO GERAL
# ============================================================================

if pagina == "🏠 Visão Geral":
    st.markdown('<div class="main-header">📊 Dashboard de Análise Exploratória - ENEM 2024</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    total_inscritos = 4332944
    total_resultados = len(df_completo)
    taxa_comparecimento = (total_resultados / total_inscritos) * 100
    ausencias = total_inscritos - total_resultados
    
    with col1:
        st.metric(
            label="📝 Total de Inscritos",
            value=f"{total_inscritos:,}".replace(',', '.')
        )
    
    with col2:
        st.metric(
            label="✅ Com Resultados",
            value=f"{total_resultados:,}".replace(',', '.')
        )
    
    with col3:
        st.metric(
            label="📊 Taxa de Comparecimento",
            value=f"{taxa_comparecimento:.2f}%"
        )
    
    with col4:
        st.metric(
            label="❌ Ausências",
            value=f"{ausencias:,}".replace(',', '.')
        )
    
    st.markdown("---")
    
    # Resumo das notas
    st.markdown('<div class="sub-header">📈 Resumo Estatístico das Notas</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📐 Matemática")
        st.metric("Média", f"{stats['nota_mt_matematica']['mean']:.2f}")
        st.metric("Desvio Padrão", f"{stats['nota_mt_matematica']['std']:.2f}")
        st.metric("Mediana", f"{stats['nota_mt_matematica']['median']:.2f}")
        st.metric("CV", f"{stats['nota_mt_matematica']['cv']:.2f}%")
    
    with col2:
        st.markdown("### ✍️ Redação")
        st.metric("Média", f"{stats['nota_redacao']['mean']:.2f}")
        st.metric("Desvio Padrão", f"{stats['nota_redacao']['std']:.2f}")
        st.metric("Mediana", f"{stats['nota_redacao']['median']:.2f}")
        st.metric("CV", f"{stats['nota_redacao']['cv']:.2f}%")
    
    with col3:
        st.markdown("### 📊 Nota Média Geral")
        st.metric("Média", f"{stats['nota_media_5_notas']['mean']:.2f}")
        st.metric("Desvio Padrão", f"{stats['nota_media_5_notas']['std']:.2f}")
        st.metric("Mediana", f"{stats['nota_media_5_notas']['median']:.2f}")
        st.metric("CV", f"{stats['nota_media_5_notas']['cv']:.2f}%")
    
    st.markdown("---")
    
    # Insights principais
    st.markdown('<div class="sub-header">💡 Principais Insights</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h4>✓ Perfil Demográfico</h4>
    <ul>
        <li>Predominância feminina: <strong>60,57%</strong> vs <strong>39,43%</strong> masculino</li>
        <li>Diferença de aproximadamente <strong>916.298</strong> inscrições</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <h4>✓ Desempenho Geral</h4>
    <ul>
        <li><strong>Redação</strong> apresenta a maior média (634,67) mas também a maior variabilidade (CV=32,95%)</li>
        <li><strong>Matemática</strong> tem o menor desempenho médio (527,08) com assimetria positiva</li>
        <li><strong>Nota Média</strong> é a mais concentrada e simétrica (CV=16,93%)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Pontos de Atenção</h4>
    <ul>
        <li>Alta taxa de ausência: <strong>30,99%</strong> dos inscritos não compareceram</li>
        <li>Matemática apresenta concentração de candidatos abaixo da média</li>
        <li>Necessário investigar desigualdades por cor/raça, renda e região</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PÁGINA 2: DISTRIBUIÇÕES DE FREQUÊNCIA
# ============================================================================

elif pagina == "📊 Distribuições de Frequência":
    st.markdown('<div class="main-header">📊 Tabelas de Distribuição de Frequência</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Seletor de variável
    variavel = st.selectbox(
        "Selecione a variável categórica:",
        ['tp_sexo', 'tp_faixa_etaria', 'tp_cor_raca', 'sg_uf_prova', 'renda_familiar'],
        format_func=lambda x: {
            'tp_sexo': 'Sexo',
            'tp_faixa_etaria': 'Faixa Etária',
            'tp_cor_raca': 'Cor/Raça',
            'sg_uf_prova': 'Estado (UF)',
            'renda_familiar': 'Renda/Escolaridade Familiar'
        }[x]
    )
    
    # Mostrar tabela
    st.markdown(f'<div class="sub-header">📋 Distribuição: {variavel.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    
    df_freq = tabelas_freq[variavel].to_pandas()
    
    # Formatar tabela
    df_freq['frequencia'] = df_freq['frequencia'].apply(lambda x: f"{x:,}".replace(',', '.'))
    df_freq['percentual'] = df_freq['percentual'].apply(lambda x: f"{x:.2f}%")
    df_freq['perc_acumulado'] = df_freq['perc_acumulado'].apply(lambda x: f"{x:.2f}%")
    
    # Renomear colunas
    df_freq = df_freq.rename(columns={
        variavel: 'Categoria',
        'frequencia': 'Frequência',
        'percentual': 'Percentual (%)',
        'perc_acumulado': 'Percentual Acumulado (%)'
    })
    
    st.dataframe(df_freq, use_container_width=True, height=400)
    
    # Estatísticas da distribuição
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Categorias", len(df_freq))
    
    with col2:
        categoria_mais_frequente = df_freq.iloc[0]['Categoria']
        st.metric("Categoria Mais Frequente", categoria_mais_frequente)
    
    with col3:
        perc_mais_frequente = df_freq.iloc[0]['Percentual (%)']
        st.metric("Percentual", perc_mais_frequente)
    
    # Download da tabela
    st.markdown("---")
    csv = df_freq.to_csv(index=False)
    st.download_button(
        label="📥 Download da Tabela (CSV)",
        data=csv,
        file_name=f"distribuicao_{variavel}.csv",
        mime="text/csv"
    )

# ============================================================================
# PÁGINA 3: VARIÁVEIS QUALITATIVAS
# ============================================================================

elif pagina == "📈 Variáveis Qualitativas":
    st.markdown('<div class="main-header">📈 Análise de Variáveis Qualitativas</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Seletor de variável
    variavel = st.selectbox(
        "Selecione a variável:",
        ['tp_sexo', 'tp_faixa_etaria', 'tp_cor_raca', 'sg_uf_prova', 'renda_familiar'],
        format_func=lambda x: {
            'tp_sexo': 'Sexo',
            'tp_faixa_etaria': 'Faixa Etária',
            'tp_cor_raca': 'Cor/Raça',
            'sg_uf_prova': 'Estado (UF)',
            'renda_familiar': 'Renda/Escolaridade Familiar'
        }[x]
    )
    
    # Tipo de gráfico
    tipo_grafico = st.radio(
        "Tipo de gráfico:",
        ["Barras Verticais", "Barras Horizontais", "Pizza"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Preparar dados
    df_freq = tabelas_freq[variavel].to_pandas()
    
    # Limitar top para UF
    if variavel == 'sg_uf_prova' and len(df_freq) > 10:
        mostrar_top = st.slider("Mostrar Top N estados:", 5, 27, 10)
        df_freq = df_freq.head(mostrar_top)
    
    # Criar gráfico
    if tipo_grafico == "Barras Verticais":
        fig = px.bar(
            df_freq,
            x=variavel,
            y='frequencia',
            title=f'Distribuição por {variavel.replace("_", " ").title()} - ENEM 2024',
            labels={variavel: variavel.replace('_', ' ').title(), 'frequencia': 'Quantidade'},
            color='frequencia',
            color_continuous_scale='Blues',
            text='frequencia'
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_xaxis(tickangle=45)
        
    elif tipo_grafico == "Barras Horizontais":
        fig = px.bar(
            df_freq,
            y=variavel,
            x='frequencia',
            orientation='h',
            title=f'Distribuição por {variavel.replace("_", " ").title()} - ENEM 2024',
            labels={variavel: variavel.replace('_', ' ').title(), 'frequencia': 'Quantidade'},
            color='frequencia',
            color_continuous_scale='Blues',
            text='frequencia'
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        
    else:  # Pizza
        fig = px.pie(
            df_freq,
            values='frequencia',
            names=variavel,
            title=f'Distribuição por {variavel.replace("_", " ").title()} - ENEM 2024',
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas
    st.markdown("---")
    st.markdown('<div class="sub-header">📊 Estatísticas da Distribuição</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Categorias", len(tabelas_freq[variavel]))
    
    with col2:
        total = df_freq['frequencia'].sum()
        st.metric("Total", f"{int(total):,}".replace(',', '.'))
    
    with col3:
        max_freq = df_freq['frequencia'].max()
        st.metric("Máx. Frequência", f"{int(max_freq):,}".replace(',', '.'))
    
    with col4:
        min_freq = df_freq['frequencia'].min()
        st.metric("Mín. Frequência", f"{int(min_freq):,}".replace(',', '.'))

# ============================================================================
# PÁGINA 4: VARIÁVEIS QUANTITATIVAS
# ============================================================================

elif pagina == "📉 Variáveis Quantitativas":
    st.markdown('<div class="main-header">📉 Análise de Variáveis Quantitativas</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Seletor de variável
    variavel = st.selectbox(
        "Selecione a variável:",
        ['nota_mt_matematica', 'nota_redacao', 'nota_media_5_notas', 'idade_calculada'],
        format_func=lambda x: {
            'nota_mt_matematica': 'Nota de Matemática',
            'nota_redacao': 'Nota de Redação',
            'nota_media_5_notas': 'Nota Média (5 provas)',
            'idade_calculada': 'Idade'
        }[x]
    )
    
    # Tabs para diferentes visualizações
    tab1, tab2, tab3 = st.tabs(["📊 Estatísticas Descritivas", "📈 Histograma", "📉 Análise de Quartis"])
    
    with tab1:
        st.markdown('<div class="sub-header">📊 Estatísticas Descritivas</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Medidas de Tendência Central")
            st.metric("Média", f"{stats[variavel]['mean']:.2f}")
            st.metric("Mediana (Q2)", f"{stats[variavel]['median']:.2f}")
            
            st.markdown("### Medidas de Dispersão")
            st.metric("Desvio Padrão", f"{stats[variavel]['std']:.2f}")
            st.metric("Coef. Variação (CV)", f"{stats[variavel]['cv']:.2f}%")
            
        with col2:
            st.markdown("### Valores Extremos")
            st.metric("Mínimo", f"{stats[variavel]['min']:.2f}")
            st.metric("Máximo", f"{stats[variavel]['max']:.2f}")
            st.metric("Amplitude", f"{stats[variavel]['max'] - stats[variavel]['min']:.2f}")
            
            st.markdown("### Quartis")
            st.metric("Q1 (25%)", f"{stats[variavel]['q1']:.2f}")
            st.metric("Q3 (75%)", f"{stats[variavel]['q3']:.2f}")
            st.metric("IQR (Q3-Q1)", f"{stats[variavel]['q3'] - stats[variavel]['q1']:.2f}")
        
        # Análise de assimetria
        st.markdown("---")
        assimetria = (stats[variavel]['mean'] - stats[variavel]['median']) / stats[variavel]['std']
        
        if abs(assimetria) < 0.1:
            tipo_assimetria = "Distribuição Simétrica"
            cor = "success-box"
        elif assimetria > 0:
            tipo_assimetria = "Assimetria Positiva (cauda à direita)"
            cor = "warning-box"
        else:
            tipo_assimetria = "Assimetria Negativa (cauda à esquerda)"
            cor = "warning-box"
        
        st.markdown(f"""
        <div class="{cor}">
        <h4>📊 Análise de Assimetria</h4>
        <p><strong>Coeficiente de Assimetria (Pearson):</strong> {assimetria:.3f}</p>
        <p><strong>Interpretação:</strong> {tipo_assimetria}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Outliers
        iqr = stats[variavel]['q3'] - stats[variavel]['q1']
        limite_inf = stats[variavel]['q1'] - 1.5 * iqr
        limite_sup = stats[variavel]['q3'] + 1.5 * iqr
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>🔍 Análise de Outliers (Método IQR)</h4>
        <p><strong>Limite Inferior:</strong> {limite_inf:.2f}</p>
        <p><strong>Limite Superior:</strong> {limite_sup:.2f}</p>
        <p><em>Valores fora destes limites são considerados outliers.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="sub-header">📈 Histograma</div>', unsafe_allow_html=True)
        
        # Controles do histograma
        col1, col2 = st.columns(2)
        with col1:
            nbins = st.slider("Número de bins:", 20, 100, 50)
        with col2:
            mostrar_estatisticas = st.checkbox("Mostrar linhas de média e mediana", value=True)
        
        # Criar histograma
        df_pandas = df_completo.to_pandas()
        
        fig = px.histogram(
            df_pandas,
            x=variavel,
            nbins=nbins,
            title=f'Distribuição: {variavel.replace("_", " ").title()} - ENEM 2024',
            labels={variavel: variavel.replace('_', ' ').title()},
            color_discrete_sequence=['#3498db'],
            marginal='box'
        )
        
        if mostrar_estatisticas:
            # Linha da média
            fig.add_vline(
                x=stats[variavel]['mean'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {stats[variavel]['mean']:.2f}",
                annotation_position="top right"
            )
            
            # Linha da mediana
            fig.add_vline(
                x=stats[variavel]['median'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"Mediana: {stats[variavel]['median']:.2f}",
                annotation_position="bottom right"
            )
        
        fig.update_layout(
            showlegend=False,
            height=600,
            yaxis_title='Frequência',
            bargap=0.05
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">📉 Análise de Quartis</div>', unsafe_allow_html=True)
        
        # Visualização de quartis
        quartis_data = pd.DataFrame({
            'Quartil': ['Mínimo', 'Q1 (25%)', 'Q2/Mediana (50%)', 'Q3 (75%)', 'Máximo'],
            'Valor': [
                stats[variavel]['min'],
                stats[variavel]['q1'],
                stats[variavel]['median'],
                stats[variavel]['q3'],
                stats[variavel]['max']
            ],
            'Percentual': ['0%', '25%', '50%', '75%', '100%']
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=quartis_data['Quartil'],
            y=quartis_data['Valor'],
            text=quartis_data['Valor'].apply(lambda x: f'{x:.2f}'),
            textposition='outside',
            marker_color=['#e74c3c', '#f39c12', '#3498db', '#f39c12', '#e74c3c']
        ))
        
        fig.update_layout(
            title=f'Distribuição de Quartis: {variavel.replace("_", " ").title()}',
            xaxis_title='Quartil',
            yaxis_title='Valor',
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretação
        st.markdown(f"""
        <div class="insight-box">
        <h4>📊 Interpretação dos Quartis</h4>
        <ul>
            <li><strong>25% dos candidatos</strong> tiraram até <strong>{stats[variavel]['q1']:.2f}</strong> pontos</li>
            <li><strong>50% dos candidatos</strong> tiraram até <strong>{stats[variavel]['median']:.2f}</strong> pontos (mediana)</li>
            <li><strong>75% dos candidatos</strong> tiraram até <strong>{stats[variavel]['q3']:.2f}</strong> pontos</li>
            <li><strong>50% centrais</strong> estão entre <strong>{stats[variavel]['q1']:.2f}</strong> e <strong>{stats[variavel]['q3']:.2f}</strong> (IQR = {stats[variavel]['q3'] - stats[variavel]['q1']:.2f})</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PÁGINA 5: BOX PLOTS
# ============================================================================

elif pagina == "📦 Box Plots":
    st.markdown('<div class="main-header">📦 Análise com Box Plots</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tipo de análise
    tipo_analise = st.radio(
        "Tipo de análise:",
        ["Comparativo entre notas", "Estratificado por variável categórica"],
        horizontal=True
    )
    
    if tipo_analise == "Comparativo entre notas":
        st.markdown('<div class="sub-header">📊 Box Plot Comparativo - Todas as Notas</div>', unsafe_allow_html=True)
        
        df_pandas = df_completo.to_pandas()
        
        # Box plot comparativo
        fig = go.Figure()
        
        cores = ['#3498db', '#e74c3c', '#2ecc71']
        nomes = ['Matemática', 'Redação', 'Média Geral']
        
        for col, cor, nome in zip(
            ['nota_mt_matematica', 'nota_redacao', 'nota_media_5_notas'],
            cores,
            nomes
        ):
            fig.add_trace(go.Box(
                y=df_pandas[col],
                name=nome,
                marker_color=cor,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title='Box Plot Comparativo - Notas ENEM 2024',
            yaxis_title='Pontuação',
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela comparativa
        st.markdown("---")
        st.markdown('<div class="sub-header">📋 Comparação de Estatísticas</div>', unsafe_allow_html=True)
        
        comparacao = pd.DataFrame({
            'Métrica': ['Q1', 'Mediana', 'Q3', 'IQR', 'Outliers Sup.'],
            'Matemática': [
                f"{stats['nota_mt_matematica']['q1']:.2f}",
                f"{stats['nota_mt_matematica']['median']:.2f}",
                f"{stats['nota_mt_matematica']['q3']:.2f}",
                f"{stats['nota_mt_matematica']['q3'] - stats['nota_mt_matematica']['q1']:.2f}",
                f"> {stats['nota_mt_matematica']['q3'] + 1.5 * (stats['nota_mt_matematica']['q3'] - stats['nota_mt_matematica']['q1']):.2f}"
            ],
            'Redação': [
                f"{stats['nota_redacao']['q1']:.2f}",
                f"{stats['nota_redacao']['median']:.2f}",
                f"{stats['nota_redacao']['q3']:.2f}",
                f"{stats['nota_redacao']['q3'] - stats['nota_redacao']['q1']:.2f}",
                f"> {min(1000, stats['nota_redacao']['q3'] + 1.5 * (stats['nota_redacao']['q3'] - stats['nota_redacao']['q1'])):.2f}"
            ],
            'Média Geral': [
                f"{stats['nota_media_5_notas']['q1']:.2f}",
                f"{stats['nota_media_5_notas']['median']:.2f}",
                f"{stats['nota_media_5_notas']['q3']:.2f}",
                f"{stats['nota_media_5_notas']['q3'] - stats['nota_media_5_notas']['q1']:.2f}",
                f"> {stats['nota_media_5_notas']['q3'] + 1.5 * (stats['nota_media_5_notas']['q3'] - stats['nota_media_5_notas']['q1']):.2f}"
            ]
        })
        
        st.dataframe(comparacao, use_container_width=True, hide_index=True)
        
    else:  # Estratificado
        st.markdown('<div class="sub-header">📊 Box Plot Estratificado</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            var_num = st.selectbox(
                "Variável numérica:",
                ['nota_mt_matematica', 'nota_redacao', 'nota_media_5_notas'],
                format_func=lambda x: {
                    'nota_mt_matematica': 'Nota de Matemática',
                    'nota_redacao': 'Nota de Redação',
                    'nota_media_5_notas': 'Nota Média (5 provas)'
                }[x]
            )
        
        with col2:
            var_cat = st.selectbox(
                "Estratificar por:",
                ['tp_sexo', 'tp_faixa_etaria', 'tp_cor_raca'],
                format_func=lambda x: {
                    'tp_sexo': 'Sexo',
                    'tp_faixa_etaria': 'Faixa Etária',
                    'tp_cor_raca': 'Cor/Raça'
                }[x]
            )
        
        df_pandas = df_completo.to_pandas()
        
        # Box plot estratificado
        fig = px.box(
            df_pandas,
            x=var_cat,
            y=var_num,
            title=f'{var_num.replace("_", " ").title()} por {var_cat.replace("_", " ").title()} - ENEM 2024',
            labels={
                var_cat: var_cat.replace('_', ' ').title(),
                var_num: var_num.replace('_', ' ').title()
            },
            color=var_cat
        )
        
        fig.update_xaxis(tickangle=45)
        fig.update_layout(showlegend=False, height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas por grupo
        st.markdown("---")
        st.markdown('<div class="sub-header">📋 Estatísticas por Grupo</div>', unsafe_allow_html=True)
        
        stats_grupo = (df_completo
            .group_by(var_cat)
            .agg([
                pl.col(var_num).mean().alias('Média'),
                pl.col(var_num).median().alias('Mediana'),
                pl.col(var_num).std().alias('Desvio Padrão'),
                pl.col(var_num).quantile(0.25).alias('Q1'),
                pl.col(var_num).quantile(0.75).alias('Q3'),
                pl.len().alias('N')
            ])
            .sort('Média', descending=True)
            .to_pandas())
        
        # Formatar números
        for col in ['Média', 'Mediana', 'Desvio Padrão', 'Q1', 'Q3']:
            stats_grupo[col] = stats_grupo[col].apply(lambda x: f"{x:.2f}")
        stats_grupo['N'] = stats_grupo['N'].apply(lambda x: f"{x:,}".replace(',', '.'))
        
        st.dataframe(stats_grupo, use_container_width=True, hide_index=True)

# ============================================================================
# PÁGINA 6: ANÁLISE DE CORRELAÇÃO
# ============================================================================

elif pagina == "🔗 Análise de Correlação":
    st.markdown('<div class="main-header">🔗 Análise de Correlação</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Calcular correlação
    df_numeric = df_completo.select([
        'idade_calculada',
        'nota_mt_matematica',
        'nota_redacao',
        'nota_media_5_notas'
    ]).to_pandas()
    
    corr_matrix = df_numeric.corr()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔥 Heatmap de Correlação", "📊 Scatter Plots", "📋 Matriz Numérica"])
    
    with tab1:
        st.markdown('<div class="sub-header">🔥 Heatmap de Correlação</div>', unsafe_allow_html=True)
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.3f',
            aspect='auto',
            title='Matriz de Correlação - ENEM 2024',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            labels=dict(color="Correlação")
        )
        
        fig.update_layout(
            height=600,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretação
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Interpretação da Correlação</h4>
        <ul>
            <li><strong>r > 0,7:</strong> Correlação forte positiva</li>
            <li><strong>0,4 < r < 0,7:</strong> Correlação moderada positiva</li>
            <li><strong>0,2 < r < 0,4:</strong> Correlação fraca positiva</li>
            <li><strong>-0,2 < r < 0,2:</strong> Correlação muito fraca/inexistente</li>
            <li><strong>r < -0,2:</strong> Correlação negativa</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="sub-header">📊 Scatter Plots</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            var_x = st.selectbox(
                "Variável X:",
                ['nota_mt_matematica', 'nota_redacao', 'nota_media_5_notas', 'idade_calculada'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            var_y = st.selectbox(
                "Variável Y:",
                ['nota_redacao', 'nota_mt_matematica', 'nota_media_5_notas', 'idade_calculada'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        # Amostra para performance
        tamanho_amostra = st.slider("Tamanho da amostra:", 1000, 50000, 10000, step=1000)
        
        df_sample = df_completo.to_pandas().sample(min(tamanho_amostra, len(df_completo)))
        
        fig = px.scatter(
            df_sample,
            x=var_x,
            y=var_y,
            title=f'Correlação: {var_x.replace("_", " ").title()} vs {var_y.replace("_", " ").title()}',
            labels={
                var_x: var_x.replace('_', ' ').title(),
                var_y: var_y.replace('_', ' ').title()
            },
            opacity=0.3,
            color='nota_media_5_notas' if 'nota_media_5_notas' not in [var_x, var_y] else None,
            color_continuous_scale='Viridis',
            trendline='ols'
        )
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar correlação
        corr_value = corr_matrix.loc[var_x, var_y]
        
        st.markdown(f"""
        <div class="{'success-box' if abs(corr_value) > 0.7 else 'warning-box' if abs(corr_value) > 0.4 else 'insight-box'}">
        <h4>📊 Coeficiente de Correlação</h4>
        <p><strong>r = {corr_value:.3f}</strong></p>
        <p><em>{'Correlação forte' if abs(corr_value) > 0.7 else 'Correlação moderada' if abs(corr_value) > 0.4 else 'Correlação fraca'}</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="sub-header">📋 Matriz de Correlação (valores numéricos)</div>', unsafe_allow_html=True)
        
        # Renomear colunas para melhor visualização
        corr_display = corr_matrix.copy()
        corr_display.index = corr_display.index.str.replace('_', ' ').str.title()
        corr_display.columns = corr_display.columns.str.replace('_', ' ').str.title()
        
        st.dataframe(
            corr_display.style.format("{:.3f}").background_gradient(cmap='RdBu_r', vmin=-1, vmax=1),
            use_container_width=True
        )
        
        # Principais correlações
        st.markdown("---")
        st.markdown('<div class="sub-header">🔝 Principais Correlações</div>', unsafe_allow_html=True)
        
        # Extrair correlações (excluindo diagonal)
        correlacoes = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlacoes.append({
                    'Variável 1': corr_matrix.columns[i].replace('_', ' ').title(),
                    'Variável 2': corr_matrix.columns[j].replace('_', ' ').title(),
                    'Correlação': corr_matrix.iloc[i, j]
                })
        
        df_corr = pd.DataFrame(correlacoes).sort_values('Correlação', ascending=False, key=abs)
        df_corr['Correlação'] = df_corr['Correlação'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(df_corr, use_container_width=True, hide_index=True)

# ============================================================================
# PÁGINA 7: ANÁLISES CRUZADAS
# ============================================================================

elif pagina == "🎯 Análises Cruzadas":
    st.markdown('<div class="main-header">🎯 Análises Cruzadas (Bivariadas)</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Seleção da análise
    tipo_cruzamento = st.selectbox(
        "Tipo de análise cruzada:",
        [
            "Média de notas por Sexo",
            "Média de notas por Cor/Raça",
            "Média de notas por Faixa Etária",
            "Média de notas por Renda Familiar"
        ]
    )
    
    # Mapear para nome da variável
    var_map = {
        "Média de notas por Sexo": "tp_sexo",
        "Média de notas por Cor/Raça": "tp_cor_raca",
        "Média de notas por Faixa Etária": "tp_faixa_etaria",
        "Média de notas por Renda Familiar": "renda_familiar"
    }
    
    var_cat = var_map[tipo_cruzamento]
    
    # Calcular médias
    medias = (df_completo
        .group_by(var_cat)
        .agg([
            pl.col('nota_mt_matematica').mean().alias('media_matematica'),
            pl.col('nota_redacao').mean().alias('media_redacao'),
            pl.col('nota_media_5_notas').mean().alias('media_geral'),
            pl.len().alias('quantidade')
        ])
        .sort('media_geral', descending=True)
        .to_pandas())
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Gráfico", "📋 Tabela"])
    
    with tab1:
        st.markdown(f'<div class="sub-header">📊 {tipo_cruzamento}</div>', unsafe_allow_html=True)
        
        # Gráfico de barras agrupadas
        fig = go.Figure()
        
        cores = ['#3498db', '#e74c3c', '#2ecc71']
        nomes = ['Matemática', 'Redação', 'Média Geral']
        
        for col, nome, cor in zip(
            ['media_matematica', 'media_redacao', 'media_geral'],
            nomes,
            cores
        ):
            fig.add_trace(go.Bar(
                name=nome,
                x=medias[var_cat],
                y=medias[col],
                text=medias[col].apply(lambda x: f'{x:.1f}'),
                textposition='auto',
                marker_color=cor
            ))
        
        fig.update_layout(
            title=tipo_cruzamento,
            xaxis_title=var_cat.replace('_', ' ').title(),
            yaxis_title='Nota Média',
            barmode='group',
            height=600,
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        melhor_grupo = medias.iloc[0][var_cat]
        melhor_media = medias.iloc[0]['media_geral']
        pior_grupo = medias.iloc[-1][var_cat]
        pior_media = medias.iloc[-1]['media_geral']
        gap = melhor_media - pior_media
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>💡 Insights</h4>
        <ul>
            <li><strong>Melhor desempenho:</strong> {melhor_grupo} ({melhor_media:.2f} pontos)</li>
            <li><strong>Menor desempenho:</strong> {pior_grupo} ({pior_media:.2f} pontos)</li>
            <li><strong>Gap:</strong> {gap:.2f} pontos ({(gap/pior_media)*100:.1f}% de diferença)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f'<div class="sub-header">📋 Tabela: {tipo_cruzamento}</div>', unsafe_allow_html=True)
        
        # Formatar tabela
        df_display = medias.copy()
        df_display['media_matematica'] = df_display['media_matematica'].apply(lambda x: f"{x:.2f}")
        df_display['media_redacao'] = df_display['media_redacao'].apply(lambda x: f"{x:.2f}")
        df_display['media_geral'] = df_display['media_geral'].apply(lambda x: f"{x:.2f}")
        df_display['quantidade'] = df_display['quantidade'].apply(lambda x: f"{x:,}".replace(',', '.'))
        
        # Renomear colunas
        df_display = df_display.rename(columns={
            var_cat: 'Categoria',
            'media_matematica': 'Média Matemática',
            'media_redacao': 'Média Redação',
            'media_geral': 'Média Geral',
            'quantidade': 'Quantidade'
        })
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Download
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download da Tabela (CSV)",
            data=csv,
            file_name=f"medias_{var_cat}.csv",
            mime="text/csv"
        )

# ============================================================================
# PÁGINA 8: RELATÓRIO COMPLETO
# ============================================================================

elif pagina == "📋 Relatório Completo":
    st.markdown('<div class="main-header">📋 Relatório Completo de Análise</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Botão de download do relatório
    if st.button("📥 Gerar e Baixar Relatório PDF"):
        st.info("⏳ Funcionalidade de geração de PDF em desenvolvimento...")
    
    st.markdown("---")
    
    # Seção 1: Resumo Executivo
    st.markdown('<div class="sub-header">1️⃣ Resumo Executivo</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Visão Geral dos Dados
    
    - **Total de Inscritos:** {4332944:,} candidatos
    - **Total com Resultados:** {len(df_completo):,} candidatos
    - **Taxa de Comparecimento:** 69,01%
    - **Ausências:** {4332944 - len(df_completo):,} candidatos (30,99%)
    
    ### Perfil Demográfico
    
    - **Sexo:** 60,57% Feminino | 39,43% Masculino
    - **Diferença:** 916.298 inscrições a mais do sexo feminino
    
    ### Desempenho Geral
    
    | Prova | Média | Mediana | Desvio Padrão | CV |
    |-------|-------|---------|---------------|-----|
    | **Matemática** | {stats['nota_mt_matematica']['mean']:.2f} | {stats['nota_mt_matematica']['median']:.2f} | {stats['nota_mt_matematica']['std']:.2f} | {stats['nota_mt_matematica']['cv']:.2f}% |
    | **Redação** | {stats['nota_redacao']['mean']:.2f} | {stats['nota_redacao']['median']:.2f} | {stats['nota_redacao']['std']:.2f} | {stats['nota_redacao']['cv']:.2f}% |
    | **Nota Média** | {stats['nota_media_5_notas']['mean']:.2f} | {stats['nota_media_5_notas']['median']:.2f} | {stats['nota_media_5_notas']['std']:.2f} | {stats['nota_media_5_notas']['cv']:.2f}% |
    """)
    
    st.markdown("---")
    
    # Seção 2: Principais Insights
    st.markdown('<div class="sub-header">2️⃣ Principais Insights</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <h4>✅ Pontos Positivos</h4>
    <ul>
        <li>Taxa de comparecimento de 69% é razoável para um exame voluntário</li>
        <li>Nota de Redação apresenta a maior média (634,67), indicando bom desempenho geral em escrita</li>
        <li>Distribuição da Nota Média é simétrica e concentrada, facilitando análises estatísticas</li>
        <li>Diversidade demográfica permite análises de equidade educacional</li>
    </ul>
    </div>
    
    <div class="warning-box">
    <h4>⚠️ Pontos de Atenção</h4>
    <ul>
        <li>Alta taxa de ausência (30,99%) - investigar causas socioeconômicas e logísticas</li>
        <li>Matemática apresenta menor média (527,08) e assimetria positiva (concentração abaixo da média)</li>
        <li>Redação com maior variabilidade (CV=32,95%) indica heterogeneidade nas habilidades de escrita</li>
        <li>Necessário investigar gaps de desempenho por cor/raça, renda e região</li>
    </ul>
    </div>
    
    <div class="insight-box">
    <h4>💡 Oportunidades de Melhoria</h4>
    <ul>
        <li>Programas de reforço em matemática para candidatos com baixo desempenho</li>
        <li>Iniciativas de letramento e escrita para reduzir dispersão em redação</li>
        <li>Políticas de inclusão para reduzir desigualdades socioeconômicas e raciais</li>
        <li>Estratégias para aumentar taxa de comparecimento (transporte, alimentação, conscientização)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Seção 3: Recomendações
    st.markdown('<div class="sub-header">3️⃣ Recomendações</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Para Gestores Públicos:
    
    1. **Equidade Educacional**
       - Implementar políticas afirmativas baseadas em dados de desempenho por grupos demográficos
       - Investir em escolas de regiões com menor desempenho
       - Criar programas de mentoria para grupos sub-representados
    
    2. **Infraestrutura de Exame**
       - Ampliar locais de prova para reduzir distâncias
       - Oferecer suporte logístico (transporte, alimentação)
       - Campanhas de conscientização sobre importância do ENEM
    
    3. **Qualidade do Ensino**
       - Reforço em matemática nas escolas públicas
       - Programas de escrita e redação
       - Formação continuada de professores
    
    ### Para Pesquisadores:
    
    1. **Análises Aprofundadas**
       - Modelagem preditiva de desempenho
       - Análise temporal (comparação com anos anteriores)
       - Estudos de equidade (interseccionalidade raça × renda × região)
    
    2. **Testes de Hipóteses**
       - Diferenças estatisticamente significativas entre grupos
       - Análise de variância (ANOVA) multifatorial
       - Regressão para identificar fatores preditivos
    
    ### Para Educadores:
    
    1. **Estratégias Pedagógicas**
       - Metodologias diferenciadas para matemática
       - Práticas intensivas de redação
       - Uso de dados para identificar alunos em risco
    
    2. **Preparação para o ENEM**
       - Simulados periódicos
       - Feedback individualizado
       - Apoio socioemocional
    """)
    
    st.markdown("---")
    
    # Seção 4: Metodologia
    st.markdown('<div class="sub-header">4️⃣ Metodologia</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Ferramentas Utilizadas
    
    - **Polars:** Processamento eficiente de dados (LazyFrame)
    - **Plotly:** Visualizações interativas
    - **Streamlit:** Interface web interativa
    - **Pandas:** Manipulação de dados para visualizações
    
    ### Técnicas Estatísticas
    
    - **Estatística Descritiva:** Média, mediana, quartis, desvio padrão, CV
    - **Análise de Distribuição:** Histogramas, assimetria, curtose
    - **Análise de Outliers:** Método IQR (1,5 × IQR)
    - **Análise de Correlação:** Coeficiente de Pearson
    - **Análise Bivariada:** Estratificação por variáveis categóricas
    
    ### Limitações
    
    - Dados de um único ano (ENEM 2024)
    - Ausência de variáveis socioeconômicas detalhadas
    - Análise exploratória (não inferencial)
    - Correlações não implicam causalidade
    """)
    
    st.markdown("---")
    
    st.success("✅ Relatório completo gerado com sucesso! Use as páginas anteriores para explorar visualizações interativas.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
    <p><strong>Dashboard de Análise Exploratória - ENEM 2024</strong></p>
    <p>Desenvolvido com ❤️ usando Streamlit, Polars e Plotly</p>
    <p><em>Dados: INEP - Microdados ENEM 2024</em></p>
</div>
""", unsafe_allow_html=True)
