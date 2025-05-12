import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Prediksi Risiko Drop Out Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Update custom CSS for professional UI with dark mode compatibility
st.markdown("""
<style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        font-size: 1.8rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    h3 {
        font-size: 1.3rem !important;
        margin-bottom: 0.6rem !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Professional Card styling */
    .metric-card {
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-bottom: 1.5rem;
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    .metric-card [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Button styling - professional */
    .stButton>button {
        background: linear-gradient(90deg, #0078d4, #0063b1);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        box-shadow: 0 4px 10px rgba(0, 120, 212, 0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #0063b1, #004a86);
        box-shadow: 0 6px 15px rgba(0, 120, 212, 0.3);
        transform: translateY(-2px);
    }
    
    /* Input fields styling */
    [data-baseweb="select"] {
        border-radius: 8px !important;
    }
    
    [data-baseweb="input"], [data-testid="stNumberInput"] div:nth-child(2) {
        border-radius: 8px !important;
    }
    
    /* Section headers - professional */
    .section-header {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #0078d4;
        background-color: rgba(128, 128, 128, 0.08);
    }
    
    /* Prediction result styling - professional */
    .prediction-container {
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        text-align: center;
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-result-header {
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Tabs styling - professional */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent !important;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #0078d4 !important;
        height: 4px !important;
        border-radius: 2px !important;
    }
    
    /* Data tables */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        border: none !important;
    }
    
    .dataframe th {
        background-color: rgba(128, 128, 128, 0.15);
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-size: 0.85rem !important;
        border: none !important;
    }
    
    .dataframe td {
        padding: 0.6rem 1rem !important;
        background-color: rgba(128, 128, 128, 0.05);
        border-bottom: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-right: none !important;
        border-left: none !important;
        font-size: 0.9rem;
    }
    
    /* Success, warning, error message styling */
    .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        padding: 1rem !important;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.01em;
        border-radius: 8px;
    }
    
    /* Selectbox styling */
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
    }
    
    /* Tooltip enhancements */
    .stTooltipIcon {
        color: #0078d4 !important;
    }
    
    /* Slider styling */
    [data-testid="stSlider"] > div {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    
    /* Application title styling */
    .app-title {
        text-align: center;
        padding: 1rem 0 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    .app-title h1 {
        margin: 0 !important;
        font-size: 2.5rem !important;
        background: linear-gradient(90deg, #0078d4, #00B294);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    
    /* Label text styling */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Chart container */
    .chart-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache functions for loading data and model
@st.cache_resource
def load_model():
    """Load the pre-trained model and feature names"""
    with open('model_dropout.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, feature_names

@st.cache_data
def load_data():
    """Load the dataset for analysis"""
    return pd.read_csv('data.csv')

# Load model and data
try:
    model, feature_names = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {str(e)}")
    st.stop()

# Define visualization functions with enhanced styling
def plot_feature_importance(model, feature_names):
    """Create feature importance plot with improved styling and dark mode compatibility"""
    feature_importances = model.named_steps['clf'].feature_importances_
    feature_names_display = [name.replace('_', ' ').title() for name in feature_names]
    
    imp_df = pd.DataFrame({
        'Feature': feature_names_display,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False).head(10)
    
    # Check if dark theme is active
    is_dark_theme = True if st.get_option("theme.base") == "dark" else False
    
    text_color = "white" if is_dark_theme else "black"
    plot_bg_color = "rgba(0,0,0,0)" if is_dark_theme else "white"
    grid_color = "rgba(255,255,255,0.1)" if is_dark_theme else "rgba(0,0,0,0.1)"
    
    # Professional color gradient
    color_scale = px.colors.sequential.Blues if not is_dark_theme else px.colors.sequential.Plasma
    
    fig = px.bar(
        imp_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='<b>Top 10 Factors Influencing Dropout Risk</b>', 
        text_auto='.2%',
        color='Importance',
        color_continuous_scale=color_scale
    )
    
    fig.update_layout(
        template="plotly_dark" if is_dark_theme else "plotly_white",
        height=500,
        font=dict(family="Inter, sans-serif", size=12, color=text_color),
        margin=dict(l=20, r=30, t=50, b=20),
        title_font=dict(size=20, family="Poppins, sans-serif", color=text_color),
        coloraxis_showscale=False,
        paper_bgcolor=plot_bg_color,
        plot_bgcolor=plot_bg_color,
        xaxis=dict(
            gridcolor=grid_color,
            title_font=dict(family="Inter, sans-serif", size=13),
            tickfont=dict(family="Inter, sans-serif", size=11),
        ),
        yaxis=dict(
            gridcolor=grid_color,
            title=None,
            tickfont=dict(family="Inter, sans-serif", size=12),
        )
    )
    
    fig.update_traces(
        textposition='auto', 
        marker_line_color='rgba(0,0,0,0)',
        marker_line_width=0,
        textfont=dict(family="Inter, sans-serif", size=12, color=text_color),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>'
    )
    
    return fig

def plot_dropout_stats(df):
    """Create enhanced pie chart for dropout statistics with dark mode support"""
    status_counts = df['Status'].value_counts()
    
    # Check if dark theme is active
    is_dark_theme = True if st.get_option("theme.base") == "dark" else False
    
    text_color = "white" if is_dark_theme else "black"
    plot_bg_color = "rgba(0,0,0,0)" if is_dark_theme else "white"
    
    # Professional color palette
    colors = ['#0078D4', '#FF8C00'] if not is_dark_theme else ['#00B7C3', '#FF5733']
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=.5,
        textinfo='percent',
        textfont=dict(family="Inter, sans-serif", size=14, color=text_color),
        marker=dict(
            colors=colors,
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': '<b>Student Status Distribution</b>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template="plotly_dark" if is_dark_theme else "plotly_white",
        font=dict(family="Inter, sans-serif", size=14, color=text_color),
        margin=dict(l=20, r=20, t=50, b=20),
        title_font=dict(size=20, family="Poppins, sans-serif", color=text_color),
        paper_bgcolor=plot_bg_color,
        plot_bgcolor=plot_bg_color,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(family="Inter, sans-serif", size=12)
        ),
        annotations=[
            dict(
                text='Student<br>Status',
                x=0.5,
                y=0.5,
                font=dict(size=16, family="Poppins, sans-serif", color=text_color),
                showarrow=False
            )
        ]
    )
    
    return fig

def plot_model_performance():
    """Create visualization of model performance metrics with dark mode support"""
    # CV Accuracy plot
    cv_scores = [0.8701, 0.8621, 0.8904, 0.8881, 0.8925]
    mean_score = np.mean(cv_scores)
    
    # Check if dark theme is active
    is_dark_theme = True if st.get_option("theme.base") == "dark" else False
    
    text_color = "white" if is_dark_theme else "black"
    plot_bg_color = "rgba(0,0,0,0)" if is_dark_theme else "white"
    grid_color = "rgba(255,255,255,0.1)" if is_dark_theme else "rgba(0,0,0,0.1)"
    
    # Professional color palette
    colors = {
        'cv': '#0078D4',
        'mean': '#FF0000',
        'precision': '#00B294',
        'recall': '#FFB900'
    }
    
    fig = go.Figure()
    
    # Add CV scores as bars
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(cv_scores))],
        y=cv_scores,
        name='CV Accuracy',
        marker_color=colors['cv'],
        hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>'
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=[f"Fold {i+1}" for i in range(len(cv_scores))],
        y=[mean_score] * len(cv_scores),
        mode='lines',
        name=f'Mean: {mean_score:.4f}',
        line=dict(color=colors['mean'], width=2, dash='dash'),
        hovertemplate=f'Mean Accuracy: {mean_score:.4f}<extra></extra>'
    ))
    
    # Add precision and recall for both classes
    fig.add_trace(go.Bar(
        x=['Non-Dropout', 'Dropout'],
        y=[0.90, 0.85],
        name='Precision',
        marker_color=colors['precision'],
        hovertemplate='<b>%{x}</b><br>Precision: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=['Non-Dropout', 'Dropout'],
        y=[0.93, 0.79],
        name='Recall',
        marker_color=colors['recall'],
        hovertemplate='<b>%{x}</b><br>Recall: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Model Performance Metrics</b>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template="plotly_dark" if is_dark_theme else "plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color=text_color),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            font=dict(family="Inter, sans-serif", size=12)
        ),
        barmode='group',
        paper_bgcolor=plot_bg_color,
        plot_bgcolor=plot_bg_color,
        xaxis=dict(
            gridcolor=grid_color,
            title=None,
            tickfont=dict(family="Inter, sans-serif", size=11)
        ),
        yaxis=dict(
            gridcolor=grid_color,
            range=[0.7, 1.0],
            tickformat='.2f',
            title=dict(
                text='Score',
                font=dict(family="Inter, sans-serif", size=13)
            ),
            tickfont=dict(family="Inter, sans-serif", size=11)
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_numeric_distribution(df, feature, title):
    """Create enhanced distribution plot for numeric features with dark mode support"""
    # Check if dark theme is active
    is_dark_theme = True if st.get_option("theme.base") == "dark" else False
    
    text_color = "white" if is_dark_theme else "black"
    plot_bg_color = "rgba(0,0,0,0)" if is_dark_theme else "white"
    grid_color = "rgba(255,255,255,0.1)" if is_dark_theme else "rgba(0,0,0,0.1)"
    
    fig = px.histogram(
        df, 
        x=feature, 
        color="Status", 
        barmode='group',
        title=title,
        color_discrete_map={"Dropout": "#ED7D31", "Graduate": "#5A9BD5"},
        opacity=0.8
    )
    
    fig.update_layout(
        template="plotly_dark" if is_dark_theme else "plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color=text_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=title,
        yaxis_title="Count",
        paper_bgcolor=plot_bg_color,
        plot_bgcolor=plot_bg_color,
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    
    return fig

def create_categorical_plot(df_melted, x, title):
    """Create enhanced bar chart for categorical features with dark mode support"""
    # Check if dark theme is active
    is_dark_theme = True if st.get_option("theme.base") == "dark" else False
    
    text_color = "white" if is_dark_theme else "black"
    plot_bg_color = "rgba(0,0,0,0)" if is_dark_theme else "white"
    grid_color = "rgba(255,255,255,0.1)" if is_dark_theme else "rgba(0,0,0,0.1)"
    
    fig = px.bar(
        df_melted, 
        x=x, 
        y="Percentage", 
        color="Status",
        title=title,
        color_discrete_map={"Dropout": "#ED7D31", "Graduate": "#5A9BD5"},
        text_auto='.1f'
    )
    
    fig.update_layout(
        template="plotly_dark" if is_dark_theme else "plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color=text_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=title,
        yaxis_title="Percentage (%)",
        paper_bgcolor=plot_bg_color,
        plot_bgcolor=plot_bg_color,
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    
    text_color_inside = "black" if is_dark_theme else "white"  # Inverted for visibility on bars
    fig.update_traces(texttemplate='%{text}%', textposition='inside', textfont=dict(color=text_color_inside))
    
    return fig

# Main application header
def render_app_header():
    """Render the application header with logo and title"""
    st.markdown('<div class="app-title">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<h1>🎓 Student Dropout Risk Prediction</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; margin-top: -0.5rem; font-size: 1.1rem; opacity: 0.8;">AI-powered early warning system for student retention</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Refactored tab functions
def render_prediction_tab():
    """Render the prediction tab content with professional styling"""
    st.markdown('<div class="section-header"><h2>Student Data Input</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="margin-bottom: 1.5rem; font-size: 1.05rem;">
    Enter student information below to predict dropout risk. The model analyzes academic performance,
    personal factors, and socioeconomic indicators to identify at-risk students.
    </p>
    """, unsafe_allow_html=True)
    
    # Create tabs for different categories of inputs to improve organization
    input_tabs = st.tabs([
        "📋 Personal & Academic", 
        "💰 Financial", 
        "📚 First Semester", 
        "🎓 Second Semester"
    ])
    
    # Tab 1: Personal & Academic Information
    with input_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header"><h3>Personal Information</h3></div>', unsafe_allow_html=True)
            age = st.number_input("Age at Enrollment", min_value=16, max_value=80, value=20, help="Age of the student when they enrolled")
            gender = st.selectbox(
                "Gender", 
                ["1", "0"], 
                format_func=lambda x: "Male" if x=="1" else "Female",
                help="Student's gender"
            )
            marital_status = st.selectbox(
                "Marital Status", 
                ["1", "2", "3", "4", "5", "6"], 
                format_func=lambda x: {
                   "1": "Single",
                   "2": "Married",
                   "3": "Widower", 
                   "4": "Divorced",
                   "5": "Facto union", 
                   "6": "Legally separated"
                }[x],
                help="Student's marital status"
            )
            international = st.selectbox(
                "International Student", 
                ["0", "1"], 
                format_func=lambda x: "Yes" if x=="1" else "No",
                help="Is the student an international student?"
            )
            special_needs = st.selectbox(
                "Special Educational Needs", 
                ["0", "1"], 
                format_func=lambda x: "Yes" if x=="1" else "No",
                help="Does the student have special educational needs?"
            )
        
        with col2:
            st.markdown('<div class="section-header"><h3>Academic Information</h3></div>', unsafe_allow_html=True)
            course = st.slider(
                "Program Code", 
                min_value=1, 
                max_value=17, 
                value=9,
                help="Numeric code representing the student's program of study"
            )
            daytime_evening = st.selectbox(
                "Course Schedule", 
                ["1", "0"], 
                format_func=lambda x: "Daytime" if x=="1" else "Evening",
                help="When classes are scheduled"
            )
            previous_qual = st.selectbox(
                "Previous Qualification", 
                [str(i) for i in range(1, 18)],
                index=0,
                help="Student's qualification before entering higher education"
            )
            admission_grade = st.slider(
                "Admission Grade", 
                min_value=100.0, 
                max_value=200.0, 
                value=120.0, 
                step=0.1,
                help="Student's admission grade (100-200 scale)"
            )
            scholarship = st.selectbox(
                "Scholarship Holder", 
                ["0", "1"], 
                format_func=lambda x: "Yes" if x=="1" else "No",
                help="Does the student have a scholarship?"
            )
    
    # Tab 2: Financial & Socioeconomic Information
    with input_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header"><h3>Financial Status</h3></div>', unsafe_allow_html=True)
            debtor = st.selectbox(
                "Debtor Status", 
                ["0", "1"], 
                format_func=lambda x: "Yes" if x=="1" else "No",
                help="Does the student have outstanding debts?"
            )
            tuition_up_to_date = st.selectbox(
                "Tuition Fees Up to Date", 
                ["0", "1"], 
                format_func=lambda x: "Yes" if x=="1" else "No",
                help="Are the student's tuition payments up to date?"
            )
        
        with col2:
            st.markdown('<div class="section-header"><h3>Socioeconomic Factors</h3></div>', unsafe_allow_html=True)
            displaced = st.selectbox(
                "Lives Far from Institution", 
                ["0", "1"], 
                format_func=lambda x: "Yes" if x=="1" else "No",
                help="Does the student live far from the institution?"
            )
            unemployment_rate = st.slider(
                "Unemployment Rate (%)", 
                min_value=7.0, 
                max_value=17.0, 
                value=10.0, 
                step=0.1,
                help="Regional unemployment rate at enrollment"
            )
            inflation_rate = st.slider(
                "Inflation Rate (%)", 
                min_value=-1.0, 
                max_value=3.0, 
                value=1.0, 
                step=0.1,
                help="Inflation rate at enrollment"
            )
            gdp = st.slider(
                "GDP Growth Rate (%)", 
                min_value=-5.0, 
                max_value=3.0, 
                value=0.0, 
                step=0.01,
                help="Gross Domestic Product growth rate at enrollment"
            )
    
    # Tab 3: First Semester Performance
    with input_tabs[2]:
        st.markdown('<div class="section-header"><h3>First Semester Performance</h3></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            units_1st_credited = st.slider(
                "Credited Units (Sem 1)", 
                min_value=0, 
                max_value=20, 
                value=0,
                help="Number of curricular units credited in the first semester"
            )
            units_1st_enrolled = st.slider(
                "Enrolled Units (Sem 1)", 
                min_value=0, 
                max_value=12, 
                value=6,
                help="Number of curricular units enrolled in the first semester"
            )
            units_1st_evaluations = st.slider(
                "Evaluated Units (Sem 1)", 
                min_value=0, 
                max_value=20, 
                value=6,
                help="Number of evaluations in the first semester"
            )
            
        with col2:
            units_1st_approved = st.slider(
                "Approved Units (Sem 1)", 
                min_value=0, 
                max_value=12, 
                value=5,
                help="Number of approved curricular units in the first semester"
            )
            units_1st_grade = st.slider(
                "Average Grade (Sem 1)", 
                min_value=0.0, 
                max_value=20.0, 
                value=12.0, 
                step=0.1,
                help="Average grade in the first semester (0-20 scale)"
            )
            units_1st_without_eval = st.slider(
                "Units Without Evaluation (Sem 1)", 
                min_value=0, 
                max_value=10, 
                value=0,
                help="Number of curricular units without evaluation in the first semester"
            )
    
    # Tab 4: Second Semester Performance
    with input_tabs[3]:
        st.markdown('<div class="section-header"><h3>Second Semester Performance</h3></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            units_2nd_credited = st.slider(
                "Credited Units (Sem 2)", 
                min_value=0, 
                max_value=20, 
                value=0,
                help="Number of curricular units credited in the second semester"
            )
            units_2nd_enrolled = st.slider(
                "Enrolled Units (Sem 2)", 
                min_value=0, 
                max_value=12, 
                value=6,
                help="Number of curricular units enrolled in the second semester"
            )
            units_2nd_evaluations = st.slider(
                "Evaluated Units (Sem 2)", 
                min_value=0, 
                max_value=20, 
                value=6,
                help="Number of evaluations in the second semester"
            )
            
        with col2:
            units_2nd_approved = st.slider(
                "Approved Units (Sem 2)", 
                min_value=0, 
                max_value=12, 
                value=5,
                help="Number of approved curricular units in the second semester"
            )
            units_2nd_grade = st.slider(
                "Average Grade (Sem 2)", 
                min_value=0.0, 
                max_value=20.0, 
                value=12.0, 
                step=0.1,
                help="Average grade in the second semester (0-20 scale)"
            )
            units_2nd_without_eval = st.slider(
                "Units Without Evaluation (Sem 2)", 
                min_value=0, 
                max_value=10, 
                value=0,
                help="Number of curricular units without evaluation in the second semester"
            )
    
    # Predict button - centered and prominent
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    predict_button = st.button("Analyze Dropout Risk", type="primary", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_button:
        # Create a dictionary with the input values
        input_data = {
            "Marital_status": float(marital_status),
            "Application_mode": 1.0,  # Default value
            "Application_order": 1.0,  # Default value
            "Course": float(course),
            "Daytime_evening_attendance": float(daytime_evening),
            "Previous_qualification": float(previous_qual),
            "Previous_qualification_grade": float(admission_grade),
            "Nacionality": 1.0,  # Default value
            "Mothers_qualification": 1.0,  # Default value
            "Fathers_qualification": 1.0,  # Default value
            "Mothers_occupation": 1.0,  # Default value
            "Fathers_occupation": 1.0,  # Default value
            "Admission_grade": float(admission_grade),
            "Displaced": float(displaced),
            "Educational_special_needs": float(special_needs),
            "Debtor": float(debtor),
            "Tuition_fees_up_to_date": float(tuition_up_to_date),
            "Gender": float(gender),
            "Scholarship_holder": float(scholarship),
            "Age_at_enrollment": float(age),
            "International": float(international),
            "Curricular_units_1st_sem_credited": float(units_1st_credited),
            "Curricular_units_1st_sem_enrolled": float(units_1st_enrolled),
            "Curricular_units_1st_sem_evaluations": float(units_1st_evaluations),
            "Curricular_units_1st_sem_approved": float(units_1st_approved),
            "Curricular_units_1st_sem_grade": float(units_1st_grade),
            "Curricular_units_1st_sem_without_evaluations": float(units_1st_without_eval),
            "Curricular_units_2nd_sem_credited": float(units_2nd_credited),
            "Curricular_units_2nd_sem_enrolled": float(units_2nd_enrolled),
            "Curricular_units_2nd_sem_evaluations": float(units_2nd_evaluations),
            "Curricular_units_2nd_sem_approved": float(units_2nd_approved),
            "Curricular_units_2nd_sem_grade": float(units_2nd_grade),
            "Curricular_units_2nd_sem_without_evaluations": float(units_2nd_without_eval),
            "Unemployment_rate": float(unemployment_rate),
            "Inflation_rate": float(inflation_rate),
            "GDP": float(gdp)
        }
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Make prediction with progress animation and professional styled container
        with st.spinner('Processing prediction...'):
            probability = model.predict_proba(input_df)[0][1]
            prediction = model.predict(input_df)[0]
        
        # Display results with enhanced professional formatting
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        
        # Risk level with theme-aware colors
        is_dark_theme = True if st.get_option("theme.base") == "dark" else False
        
        # Define text_color based on the theme - fix for NameError
        text_color = "white" if is_dark_theme else "black"
        
        if probability < 0.3:
            risk_level = "Low"
            color = "#00CC96" if is_dark_theme else "green"
            gauge_color = "#00CC96"
        elif probability < 0.7:
            risk_level = "Medium"
            color = "#FFA15A" if is_dark_theme else "orange"
            gauge_color = "#FFA15A"
        else:
            risk_level = "High"
            color = "#EF553B" if is_dark_theme else "red"
            gauge_color = "#EF553B"
        
        # Add professional header for prediction
        st.markdown('<div class="prediction-result-header">', unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; margin-bottom: 0.5rem;'>Analysis Results</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: {color}; margin-top: 0;'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a gauge chart for probability
        gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%", 'font': {'size': 26, 'family': 'Inter'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': text_color},
                'bar': {'color': gauge_color},
                'bgcolor': "rgba(255, 255, 255, 0.1)",
                'borderwidth': 2,
                'bordercolor': "rgba(255, 255, 255, 0.3)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 204, 150, 0.3)'},
                    {'range': [30, 70], 'color': 'rgba(255, 161, 90, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(239, 85, 59, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        ))
        
        gauge.update_layout(
            title="Dropout Probability",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=50, b=20),
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(gauge, use_container_width=True)
        
        # Display status prediction
        status = "Dropout" if prediction == 1 else "Graduate"
        icon = "🚫" if prediction == 1 else "🎓"
        st.markdown(f"<h4 style='text-align: center; margin-bottom: 1.5rem;'>Predicted Status: {icon} {status}</h4>", unsafe_allow_html=True)
        
        # Add professional recommendations section
        st.markdown('<div class="section-header" style="margin-top: 1rem;"><h3>Recommended Interventions</h3></div>', unsafe_allow_html=True)
        
        if risk_level == "Low":
            st.success("""
            ✅ **Student shows low dropout risk.**
            
            **Recommended actions:**
            - Maintain current academic support systems
            - Continue periodic monitoring
            - Encourage participation in extracurricular activities
            - Offer advanced learning opportunities to maintain engagement
            """)
        elif risk_level == "Medium":
            st.warning("""
            ⚠️ **Student shows moderate dropout risk.**
            
            **Recommended actions:**
            - Implement more intensive academic monitoring
            - Schedule regular academic counseling and mentoring
            - Evaluate contributing risk factors and develop mitigation plans
            - Provide additional support for challenging courses
            - Consider peer mentoring programs
            """)
        else:
            st.error("""
            🚨 **Student shows high dropout risk!**
            
            **Recommended actions:**
            - Immediate structured academic intervention required
            - Create personalized support plan with academic advisor
            - Consider financial aid/scholarship opportunities
            - Schedule weekly progress monitoring sessions
            - Evaluate and adjust academic workload if necessary
            - Connect with counseling services for non-academic challenges
            """)
        
        # Add a collapsible section with key factors
        with st.expander("View Key Risk Factors"):
            # Create a radar chart of important features for this student
            top_features = ["Age_at_enrollment", "Curricular_units_1st_sem_approved", 
                           "Curricular_units_1st_sem_grade", "Scholarship_holder", "Debtor"]
            feature_values = [input_data[f] for f in top_features]
            
            # Normalize values for radar chart
            normalized_values = []
            for i, f in enumerate(top_features):
                if f == "Age_at_enrollment":
                    # Age: normalize between 18-40
                    normalized_values.append(min(max((feature_values[i] - 18) / (40 - 18), 0), 1))
                elif f == "Curricular_units_1st_sem_grade":
                    # Grades: normalize between 0-20
                    normalized_values.append(feature_values[i] / 20)
                elif f == "Curricular_units_1st_sem_approved":
                    # Approved units: normalize between 0-12
                    normalized_values.append(feature_values[i] / 12)
                else:
                    # Binary features: already 0-1
                    normalized_values.append(feature_values[i])
            
            # Display readable feature names
            readable_features = [
                "Age", "Approved Units (Sem 1)", 
                "Grade Average (Sem 1)", "Scholarship Status", "Debtor Status"
            ]
            
            radar = go.Figure()
            
            radar.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=readable_features,
                fill='toself',
                name='Student',
                line_color='#0078D4',
                fillcolor='rgba(0, 120, 212, 0.3)'
            ))
            
            radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                title="Key Factors Profile",
                font=dict(family="Inter, sans-serif"),
                height=350,
                margin=dict(l=80, r=80, t=50, b=50),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(radar, use_container_width=True)
            
            st.markdown(f"""
            **Top factors affecting this student's risk level:**
            
            1. **First semester performance**: {units_1st_approved} of {units_1st_enrolled} units approved
            2. **Average grade in first semester**: {units_1st_grade}/20
            3. **Financial status**: {"Has debt" if debtor == "1" else "No debt"}, {"Tuition up to date" if tuition_up_to_date == "1" else "Tuition not up to date"}
            4. **Scholarship status**: {"Has scholarship" if scholarship == "1" else "No scholarship"}
            5. **Course schedule**: {"Daytime" if daytime_evening == "1" else "Evening"} classes
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_statistics_tab():
    """Render the statistics tab content with professional styling"""
    st.markdown('<div class="section-header"><h2>Dataset Analysis</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="margin-bottom: 1.5rem; font-size: 1.05rem;">
    This dashboard visualizes key statistics from the student dataset, highlighting factors that influence dropout rates
    and demonstrating model performance metrics.
    </p>
    """, unsafe_allow_html=True)
    
    # Dashboard-style overview with metrics
    st.markdown('<h3>Dataset Overview</h3>', unsafe_allow_html=True)
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Students", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats2:
        dropout_count = (df['Status'] == 'Dropout').sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dropout Count", f"{dropout_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats3:
        dropout_rate = (df['Status'] == 'Dropout').mean() * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dropout Rate", f"{dropout_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats4:
        graduate_rate = (df['Status'] == 'Graduate').mean() * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Graduation Rate", f"{graduate_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced visualizations
    st.markdown('<div class="section-header"><h3>Data Visualizations</h3></div>', unsafe_allow_html=True)
    viz_tabs = st.tabs(["Student Status", "Influential Factors", "Model Performance"])
    
    with viz_tabs[0]:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(plot_dropout_stats(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with viz_tabs[1]:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(plot_feature_importance(model, feature_names), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with viz_tabs[2]:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(plot_model_performance(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("View Detailed Model Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Cross-Validation Scores")
                cv_data = pd.DataFrame({
                    'Fold': [1, 2, 3, 4, 5],
                    'Accuracy': [0.8701, 0.8621, 0.8904, 0.8881, 0.8925]
                })
                st.dataframe(cv_data, use_container_width=True)
                st.markdown(f"**Mean Accuracy**: 0.8807 (88.07%)")
            
            with col2:
                st.markdown("#### Classification Report")
                report_data = pd.DataFrame({
                    'Class': ['Non-Dropout', 'Dropout'],
                    'Precision': [0.90, 0.85],
                    'Recall': [0.93, 0.79],
                    'F1-Score': [0.92, 0.82],
                    'Support': [901, 427]
                })
                st.dataframe(report_data, use_container_width=True)
    
    # Deep dive analysis with factor selector
    st.markdown('<div class="section-header"><h3>Factor Analysis</h3></div>', unsafe_allow_html=True)
    
    # Create dataframe with dropout flag for analysis
    df_analysis = df.copy()
    df_analysis['Dropout'] = (df_analysis['Status'] == 'Dropout').astype(int)
    
    # Add a more intuitive factor selector
    factor_mapping = {
        "Age_at_enrollment": "Age at Enrollment",
        "Curricular_units_1st_sem_approved": "Approved Units (Sem 1)",
        "Curricular_units_1st_sem_grade": "Average Grade (Sem 1)",
        "Curricular_units_2nd_sem_approved": "Approved Units (Sem 2)",
        "Debtor": "Debtor Status",
        "Tuition_fees_up_to_date": "Tuition Up to Date",
        "Scholarship_holder": "Scholarship Holder",
        "Gender": "Gender"
    }
    
    reverse_mapping = {v: k for k, v in factor_mapping.items()}
    
    selected_factor_display = st.selectbox(
        "Select Factor to Analyze",
        options=list(factor_mapping.values())
    )
    selected_factor = reverse_mapping[selected_factor_display]
    
    # Display different charts based on the type of factor
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    if selected_factor in ["Age_at_enrollment", "Curricular_units_1st_sem_approved", 
                        "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_approved"]:
        fig = create_numeric_distribution(df, selected_factor, f"Distribution of {selected_factor_display} by Status")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics in cards
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"<h4>Statistics for Dropout Students</h4>", unsafe_allow_html=True)
            dropout_stats = df[df['Status'] == 'Dropout'][selected_factor].describe()
            st.write(f"**Mean:** {dropout_stats['mean']:.2f}")
            st.write(f"**Median:** {dropout_stats['50%']:.2f}")
            st.write(f"**Std Dev:** {dropout_stats['std']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_sum2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"<h4>Statistics for Graduate Students</h4>", unsafe_allow_html=True)
            graduate_stats = df[df['Status'] == 'Graduate'][selected_factor].describe()
            st.write(f"**Mean:** {graduate_stats['mean']:.2f}")
            st.write(f"**Median:** {graduate_stats['50%']:.2f}")
            st.write(f"**Std Dev:** {graduate_stats['std']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # For categorical features
        cross_tab = pd.crosstab(
            df_analysis[selected_factor], df_analysis["Status"], 
            normalize='index'
        ).reset_index()
        
        cross_tab_melted = pd.melt(
            cross_tab, id_vars=[selected_factor], 
            value_vars=["Dropout", "Graduate"], 
            var_name="Status", value_name="Percentage"
        )
        cross_tab_melted["Percentage"] = cross_tab_melted["Percentage"] * 100
        
        fig = create_categorical_plot(cross_tab_melted, selected_factor, f"Status Distribution by {selected_factor_display}")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add table view of data with professional styling
    st.markdown(f"<h4>Detailed Breakdown by {selected_factor_display}</h4>", unsafe_allow_html=True)
    
    # Format table for better presentation
    formatted_table = pd.crosstab(
        df_analysis[selected_factor], df_analysis["Status"],
        margins=True, margins_name="Total"
    ).reset_index()
    
    # Map binary features to readable values
    if selected_factor in ["Debtor", "Tuition_fees_up_to_date", "Scholarship_holder", "Gender"]:
        if selected_factor == "Gender":
            mapping = {0: "Female", 1: "Male"}
        else:
            mapping = {0: "No", 1: "Yes"}
        formatted_table[selected_factor] = formatted_table[selected_factor].map(
            lambda x: mapping.get(x, x)
        )
    
    st.dataframe(formatted_table, use_container_width=True)
    
    # Add insights section
    st.markdown('<div class="section-header"><h3>Key Insights</h3></div>', unsafe_allow_html=True)
    
    if selected_factor == "Curricular_units_1st_sem_approved":
        st.info("""
        📊 **First semester performance is highly predictive of dropout risk.**
        
        Students who pass fewer courses in their first semester show significantly higher dropout rates. 
        Early academic intervention for struggling students can substantially improve retention.
        """)
    elif selected_factor == "Scholarship_holder":
        st.info("""
        💰 **Scholarship status correlates with graduation rates.**
        
        Students with scholarships show lower dropout rates, suggesting that financial support 
        plays an important role in student retention and success.
        """)
    elif selected_factor == "Debtor":
        st.info("""
        💳 **Students with outstanding debts are at higher risk of dropping out.**
        
        Financial difficulties appear to be a significant factor in dropout decisions.
        Financial counseling and flexible payment options may help reduce dropout rates.
        """)
    else:
        st.info(f"""
        🔍 **Analyze how {selected_factor_display} relates to dropout risk.**
        
        The charts above show the relationship between this factor and student outcomes.
        Use this information to identify at-risk students and design appropriate interventions.
        """)

def render_about_tab():
    """Render the about tab content with professional styling"""
    st.markdown('<div class="section-header"><h2>About This Application</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        ### Student Dropout Risk Prediction System
        
        This application uses machine learning (XGBoost algorithm) to predict the likelihood of a student 
        dropping out based on academic, demographic, and socioeconomic factors. The model analyzes patterns 
        in historical student data to identify at-risk individuals before they drop out.
        
        #### How to Use This Tool:
        1. Navigate to the **Prediction** tab to input student information
        2. Complete the form with available student data across all categories
        3. Click the **Analyze Dropout Risk** button to generate a prediction
        4. Review the risk assessment and recommended interventions
        
        #### Key Features:
        - High-accuracy dropout risk prediction (88% accuracy)
        - Comprehensive data visualization and analysis
        - Factor-specific risk assessment and statistics
        - Evidence-based intervention recommendations
        - User-friendly interface with detailed explanations
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2452/2452279.png", width=200)
    
    # Model details with professional styling
    st.markdown('<div class="section-header"><h3>The Prediction Model</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Model Architecture
        
        The system uses a **Gradient Boosting** model (XGBoost), which combines multiple decision trees 
        to create a powerful ensemble classifier.
        
        **Development Process:**
        - Feature engineering and selection from student records
        - Data preprocessing including normalization and encoding
        - Class imbalance handling with SMOTE technique
        - Hyperparameter optimization using grid search
        - 5-fold cross-validation for robust performance estimation
        """)
        
    with col2:
        st.markdown("""
        #### Performance Metrics
        
        The model demonstrates strong predictive power with:
        
        - **Overall Accuracy**: 88.07%
        - **Precision (Dropout)**: 85%
        - **Recall (Dropout)**: 79%
        - **F1-Score (Dropout)**: 82%
        
        The model is specifically tuned to minimize false negatives, ensuring 
        that at-risk students are not overlooked.
        """)
    
    # Dataset information in an expander
    with st.expander("Dataset Information"):
        st.markdown("""
        #### About the Dataset
        
        The dataset contains academic records of higher education students, including:
        
        - Demographic details (age, gender, marital status)
        - Academic performance metrics across semesters
        - Socioeconomic indicators
        - Financial status (debts, scholarship status)
        - Course and enrollment information
        
        The data was collected over multiple academic years, with careful preprocessing 
        to ensure privacy and representativeness.
        """)
    
    # Implementation details in an expander
    with st.expander("Technical Implementation"):
        st.markdown("""
        #### Technology Stack
        
        This application is built with:
        
        - **Python** as the core programming language
        - **Streamlit** for the web interface
        - **Scikit-learn** and **XGBoost** for machine learning
        - **Pandas** for data manipulation
        - **Plotly** for interactive visualizations
        
        The application is designed to be scalable and can be integrated with 
        student information systems for automated risk assessment.
        """)
    
    # References and disclaimer
    st.markdown('<div class="section-header"><h3>References & Disclaimer</h3></div>', unsafe_allow_html=True)
    
    st.markdown("""
    #### Research Basis
    
    This project is based on educational data mining research and best practices 
    in student retention. Key references include studies on early warning systems 
    and academic performance prediction models in higher education.
    """)
    
    st.info("""
    **Disclaimer**: This tool is designed to support educational decision-making, not replace it. 
    Predictions should be interpreted as risk indicators that warrant further investigation by 
    qualified education professionals. Always consider the full context of a student's situation 
    when making intervention decisions.
    """)
    
    # Developers section
    st.markdown("#### Development Team")
    st.write("This application was developed as part of the IDCamp Machine Learning Track final project.")

# Main app structure - Create tabs and call the refactored functions
render_app_header()
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dataset Statistics", "ℹ️ About"])

with tab1:
    render_prediction_tab()

with tab2:
    render_statistics_tab()

with tab3:
    render_about_tab()
