import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="SkyGuard AI - AQI Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced UI ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background Gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .st-emotion-cache-1d391kg {
        color: white;
    }
    
    /* Card-like containers */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e7ff;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
    }
    
    h2 {
        color: #1e3c72;
        font-weight: 600;
    }
    
    h3 {
        color: #2a5298;
        font-weight: 600;
    }
    
    /* AQI Result Card */
    .aqi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5);
        margin: 2rem 0;
    }
    
    .aqi-value {
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .aqi-label {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: #e8e8e8;
        padding: 1rem 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        background-color: transparent;
        border-radius: 10px;
        transition: all 0.3s ease;
        color: #333;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- AQI Color & Health Info (CPCB / Indian National AQI scale) ---
# NOTE: This app is trained on CPCB station data, whose AQI_Bucket labels are
# Good / Satisfactory / Moderate / Poor / Very Poor / Severe (NOT the US EPA
# scale). Keys here match le.classes_ exactly so the model's predicted label
# can be looked up directly - no separate/duplicate categorization logic.
AQI_CATEGORY_INFO = {
    'Good': {
        'range': (0, 50),
        'color': '#00E400',
        'icon': '😊',
        'health': 'Minimal impact. Air quality poses little or no risk.',
        'recommendation': 'Great day to enjoy outdoor activities!'
    },
    'Satisfactory': {
        'range': (51, 100),
        'color': '#A3C853',
        'icon': '🙂',
        'health': 'Minor breathing discomfort may occur for sensitive people.',
        'recommendation': 'Outdoor activity is generally fine for most people.'
    },
    'Moderate': {
        'range': (101, 200),
        'color': '#FFFF00',
        'icon': '😐',
        'health': 'Breathing discomfort possible for people with lung disease, asthma, children, and older adults.',
        'recommendation': 'Sensitive groups should limit prolonged outdoor exertion.'
    },
    'Poor': {
        'range': (201, 300),
        'color': '#FF7E00',
        'icon': '😷',
        'health': 'Breathing discomfort on prolonged exposure for most people.',
        'recommendation': 'Reduce prolonged or heavy outdoor exertion, especially sensitive groups.'
    },
    'Very Poor': {
        'range': (301, 400),
        'color': '#FF0000',
        'icon': '😨',
        'health': 'Respiratory illness risk on prolonged exposure.',
        'recommendation': 'Avoid outdoor exertion; sensitive groups should stay indoors.'
    },
    'Severe': {
        'range': (401, 500),
        'color': '#8B0000',
        'icon': '☠️',
        'health': 'Serious risk, affecting even healthy people, with more serious effects on those with existing disease.',
        'recommendation': 'Everyone should avoid outdoor exertion and stay indoors.'
    },
}


def get_aqi_info(category_label):
    """Look up display info for a category label returned by the model.
    Falls back gracefully if the model ever returns an unseen label."""
    return AQI_CATEGORY_INFO.get(category_label, {
        'range': (None, None),
        'color': '#808080',
        'icon': '❓',
        'health': 'Unrecognized category returned by the model.',
        'recommendation': 'Please check the model / label encoder.'
    })


def pm25_subindex(pm25_value):
    """CPCB PM2.5 sub-index (0-500), for display as an auxiliary indicative
    number alongside the model's categorical prediction. This is NOT the
    model's output - it's a single-pollutant estimate shown for context."""
    breakpoints = [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ]
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25_value <= bp_hi:
            return round(((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25_value - bp_lo) + aqi_lo)
    return 500

# --- Load Model & Helpers ---
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(base_dir, 'models/aqi_decision_tree.pkl'))
        le = joblib.load(os.path.join(base_dir, 'models/label_encoder.pkl'))
        medians = joblib.load(os.path.join(base_dir, 'models/medians.pkl'))

        # Feature order the model was actually fit with. Prefer the explicit
        # feature_order.pkl saved by train_model.py (if present); otherwise
        # fall back to sklearn's auto-stored feature_names_in_, which is set
        # automatically whenever .fit() is called with a DataFrame - as is
        # already the case for the current model. Either way this avoids a
        # hand-typed column list that could silently drift out of sync with
        # train_model.py.
        feature_order_path = os.path.join(base_dir, 'models/feature_order.pkl')
        if os.path.exists(feature_order_path):
            feature_order = joblib.load(feature_order_path)
        else:
            feature_order = list(getattr(model, 'feature_names_in_', medians.keys()))

        return model, le, medians, feature_order, None
    except FileNotFoundError as e:
        return None, None, None, None, f"Model files not found: {e}. Run train_model.py first."
    except Exception as e:
        return None, None, None, None, f"Failed to load model artifacts: {e}"

model, le, medians, FEATURE_ORDER, load_error = load_artifacts()

# --- Header Navigation ---
st.markdown("# 🌍 SkyGuard AI")
st.markdown("### Advanced Air Quality Intelligence Platform")
st.markdown("---")

# Horizontal Navigation Tabs
page = st.tabs(["🔮 Prediction Portal", "📊 Analytics Hub", "ℹ️ About"])


# TAB 1: PREDICTION PORTAL

with page[0]:
    st.title("🔮 Air Quality Prediction")
    st.markdown("### Enter environmental parameters for real-time AQI analysis")
    
    if model is None:
        st.error(f"⚠️ {load_error}")
    else:
        # Input Section
        st.markdown("---")
        st.subheader("📊 Environmental Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Particulate Matter**")
            pm25 = st.number_input("PM2.5 (µg/m³)", value=float(medians['PM2.5']), min_value=0.0, step=1.0, help="Fine particles < 2.5 micrometers")
            pm10 = st.number_input("PM10 (µg/m³)", value=float(medians['PM10']), min_value=0.0, step=1.0, help="Particles < 10 micrometers")
        
        with col2:
            st.markdown("**Gaseous Pollutants**")
            no2 = st.number_input("NO2 (µg/m³)", value=float(medians['NO2']), min_value=0.0, step=1.0, help="Nitrogen Dioxide")
            co = st.number_input("CO (mg/m³)", value=float(medians['CO']), min_value=0.0, step=0.1, help="Carbon Monoxide")
            nh3 = st.number_input("NH3 (µg/m³)", value=float(medians['NH3']), min_value=0.0, step=1.0, help="Ammonia")
        
        with col3:
            st.markdown("**Additional Parameters**")
            o3 = st.number_input("O3 (µg/m³)", value=float(medians['O3']), min_value=0.0, step=1.0, help="Ozone")
            d = st.date_input("Analysis Date", datetime.date.today())
            t = st.time_input("Analysis Time", datetime.datetime.now().time())
        
        st.markdown("---")
        
        # Prediction Button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_btn = st.button("🚀 Generate Prediction", type="primary", use_container_width=True)
        
        if predict_btn:
            with st.spinner("🔄 Analyzing environmental data..."):
                # Assemble the feature dict for this input. Fields not
                # collected from the user fall back to training-set medians.
                feature_values = {
                    'PM2.5': pm25, 'PM10': pm10,
                    'NO': medians.get('NO'), 'NO2': no2, 'NOx': medians.get('NOx'),
                    'NH3': nh3, 'CO': co, 'SO2': medians.get('SO2'), 'O3': o3,
                    'Benzene': medians.get('Benzene'), 'Toluene': medians.get('Toluene'),
                    'Year': d.year, 'Month': d.month, 'Day': d.day, 'Hour': t.hour,
                }
                # Build the DataFrame in the EXACT column order the model was
                # trained on (model.feature_names_in_), rather than a
                # hand-typed list that could silently drift out of sync.
                input_df = pd.DataFrame([[feature_values[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)

                # Get the model's actual prediction - this IS the ML output,
                # and it now drives everything shown below.
                res = model.predict(input_df)[0]
                label = le.inverse_transform([res])[0]
                aqi_info = get_aqi_info(label)

                # Indicative numeric sub-index from PM2.5 alone (CPCB scale).
                # Shown as supporting context, NOT as if it were the model's
                # output - it only looks at one of the model's 15 inputs.
                pm25_index = pm25_subindex(pm25)

                # Display Results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")

                # Category Card - driven by the model's predicted label
                st.markdown(f"""
                <div class="aqi-card" style="background: linear-gradient(135deg, {aqi_info['color']}88 0%, {aqi_info['color']}cc 100%);">
                    <div style="font-size: 3rem;">{aqi_info['icon']}</div>
                    <div class="aqi-label">{label}</div>
                    <div style="font-size: 1.2rem; opacity: 0.9;">Predicted AQI Category (Decision Tree model)</div>
                </div>
                """, unsafe_allow_html=True)

                st.caption(f"📌 Indicative PM2.5 sub-index (single-pollutant estimate, CPCB scale): **{pm25_index}** — shown for context, not the model's prediction.")
                
                # Details in columns
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"""
                    <div class="info-box">
                        <h3>🏥 Health Impact</h3>
                        <p>{aqi_info['health']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class="info-box">
                        <h3>💡 Recommendation</h3>
                        <p>{aqi_info['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Pollutant Breakdown
                st.markdown("### 📈 Pollutant Analysis")
                pollutant_data = {
                    'Pollutant': ['PM2.5', 'PM10', 'NO2', 'CO', 'NH3', 'O3'],
                    'Value': [pm25, pm10, no2, co, nh3, o3],
                    'Unit': ['µg/m³', 'µg/m³', 'µg/m³', 'mg/m³', 'µg/m³', 'µg/m³']
                }
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=pollutant_data['Pollutant'],
                        y=pollutant_data['Value'],
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'],
                        text=pollutant_data['Value'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Current Pollutant Levels",
                    xaxis_title="Pollutant Type",
                    yaxis_title="Concentration",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)


# TAB 2: ANALYTICS HUB

with page[1]:
    st.title("📊 Environmental Data Intelligence")
    st.markdown("### Comprehensive analysis of historical air quality patterns")
    
    @st.cache_data
    def get_data():
        return pd.read_csv("data/station_hour.csv", low_memory=False).dropna(subset=['AQI'])
    
    try:
        df = get_data()
        
        # --- KPI METRICS ---
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        
        avg_aqi = round(df['AQI'].mean(), 1)
        peak_pm25 = round(df['PM2.5'].max(), 1)
        data_points = f"{len(df)//1000}K"
        max_aqi = round(df['AQI'].max(), 1)
        
        with m1:
            st.metric(
                label="📊 Average AQI",
                value=avg_aqi,
                delta=f"{round(df['AQI'].std(), 1)} std dev"
            )
        
        with m2:
            st.metric(
                label="🔴 Peak PM2.5",
                value=f"{peak_pm25} µg/m³",
                delta="Critical Pollutant"
            )
        
        with m3:
            st.metric(
                label="📈 Data Points",
                value=data_points,
                delta="Analyzed"
            )
        
        with m4:
            st.metric(
                label="⚠️ Max AQI",
                value=max_aqi,
                delta="Historical Peak"
            )
        
        st.markdown("---")
        
        # --- TABS ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎨 AQI Distribution", 
            "🧪 Pollutant Mix", 
            "⏰ Time Patterns",
            "🗺️ Heatmap Analysis"
        ])
        
        with tab1:
            st.subheader("Air Quality Category Distribution")
            st.markdown("Understanding the frequency of different air quality levels")
            
            col_chart1, col_chart2 = st.columns([2, 1])
            
            with col_chart1:
                fig = px.histogram(
                    df, 
                    x="AQI_Bucket", 
                    color="AQI_Bucket",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title="AQI Category Frequency"
                )
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_chart2:
                bucket_counts = df['AQI_Bucket'].value_counts()
                fig2 = px.pie(
                    values=bucket_counts.values,
                    names=bucket_counts.index,
                    hole=0.5,
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    title="Proportion"
                )
                fig2.update_layout(height=450)
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("🧪 Average Pollutant Composition")
            st.markdown("Breakdown of major air pollutants contributing to overall air quality")
            
            pollutants = ['PM2.5', 'PM10', 'NO2', 'NH3', 'CO', 'SO2', 'O3']
            avg_values = df[pollutants].mean()
            
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                fig = px.pie(
                    values=avg_values,
                    names=pollutants,
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    title="Pollutant Distribution (by average concentration)"
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_p2:
                fig2 = go.Figure(data=[
                    go.Bar(
                        y=pollutants,
                        x=avg_values.values,
                        orientation='h',
                        marker=dict(
                            color=avg_values.values,
                            colorscale='Viridis',
                            showscale=True
                        )
                    )
                ])
                fig2.update_layout(
                    title="Average Concentration Levels",
                    xaxis_title="Concentration",
                    yaxis_title="Pollutant",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("⏰ Temporal Patterns in Air Quality")
            st.markdown("Discover how AQI varies throughout the day and across months")
            
            df['Hour'] = pd.to_datetime(df['Datetime']).dt.hour
            df['Month'] = pd.to_datetime(df['Datetime']).dt.month
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                hourly_aqi = df.groupby('Hour')['AQI'].mean().reset_index()
                fig = px.line(
                    hourly_aqi,
                    x="Hour",
                    y="AQI",
                    markers=True,
                    title="Daily AQI Cycle (24-Hour Pattern)"
                )
                fig.update_traces(
                    line_color='#667eea',
                    line_width=3,
                    marker=dict(size=8)
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_t2:
                monthly_aqi = df.groupby('Month')['AQI'].mean().reset_index()
                fig2 = px.area(
                    monthly_aqi,
                    x="Month",
                    y="AQI",
                    title="Monthly AQI Trends"
                )
                fig2.update_traces(
                    fillcolor='rgba(102, 126, 234, 0.3)',
                    line_color='#764ba2',
                    line_width=3
                )
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab4:
            st.subheader("🗺️ Correlation Heatmap")
            st.markdown("Understand relationships between different pollutants")
            
            corr_cols = ['PM2.5', 'PM10', 'NO2', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
            corr_matrix = df[corr_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_cols,
                y=corr_cols,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Pollutant Correlation Matrix"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Insight**: Strong positive correlations (red) indicate pollutants that tend to increase together, suggesting common sources or atmospheric conditions.")
    
    except FileNotFoundError:
        st.error("⚠️ Data file not found. Please ensure 'data/station_hour.csv' exists.")


# TAB 3: ABOUT

with page[2]:
    st.title("ℹ️ About SkyGuard AI")
    
    st.markdown("""
    ### 🌍 Mission
    SkyGuard AI is an advanced air quality monitoring and prediction system designed to help communities 
    make informed decisions about outdoor activities and health protection.
    
    ---
    
    ### 🎯 Features
    
    **🔮 AI-Powered Predictions**
    - Real-time AQI classification using machine learning
    - Numerical AQI calculation based on EPA standards
    - Health recommendations based on air quality levels
    
    **📊 Comprehensive Analytics**
    - Historical data analysis and visualization
    - Pollutant composition breakdown
    - Temporal pattern recognition
    - Correlation analysis between pollutants
    
    **💡 Health Guidance**
    - Personalized recommendations for different AQI levels
    - Risk assessment for sensitive groups
    - Outdoor activity planning assistance
    
    ---
    
    ### 📈 Understanding AQI
    
    This app predicts the **CPCB National AQI category** (the scale used for Indian station data), divided into six bands:
    """)
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        **😊 Good (0-50)**  
        Minimal impact

        **🙂 Satisfactory (51-100)**  
        Minor discomfort for sensitive people
        """)
    
    with col_info2:
        st.markdown("""
        **😐 Moderate (101-200)**  
        Discomfort for sensitive groups

        **😷 Poor (201-300)**  
        Discomfort on prolonged exposure
        """)
    
    with col_info3:
        st.markdown("""
        **😨 Very Poor (301-400)**  
        Respiratory illness risk

        **☠️ Severe (401-500)**  
        Serious risk, affects healthy people too
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🛠️ Technology Stack
    
    - **Machine Learning**: Decision Tree Classifier with scikit-learn
    - **Visualization**: Plotly for interactive charts
    - **Framework**: Streamlit for web interface
    - **Data Processing**: Pandas & NumPy
    
    ---
    
    ### 📝 Data Sources
    
    The model is trained on comprehensive hourly air quality data including:
    - Particulate Matter (PM2.5, PM10)
    - Gaseous Pollutants (NO2, CO, NH3, O3, SO2)
    - Volatile Organic Compounds (Benzene, Toluene)
    - Temporal features for pattern recognition
    
    ---
    
    ### 👨‍💻 Development
    
    Built with ❤️ for environmental awareness and public health protection.
    
    """)
    
    st.success("🌱 Together, we can breathe easier!")
