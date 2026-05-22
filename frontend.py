"""
Equip-GuardianAngel - Professional Frontend
Connects to FastAPI Backend
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
import os

API_URL = os.getenv("API_URL", "http://host.docker.internal:8000/api/v1/predict")


st.set_page_config(
    page_title="Equip-GuardianAngel",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed" 
)


st.markdown("""
    <style>
        /* Background and general styling */
        .main {
            background-color: #0a101d;
            color: #e2e8f0;
        }
        
        .stApp {
            background-color: #0a101d;
        }
        
        /* Metric styling */
        .metric-card {
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #3b82f6;
            color: white;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-healthy {
            border-left-color: #10b981;
            background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        }
        
        .metric-warning {
            border-left-color: #f59e0b;
            background: linear-gradient(135deg, #78350f 0%, #d97706 100%);
        }
        
        .metric-critical {
            border-left-color: #ef4444;
            background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        }
        
        /* Button styling */
.stButton > button {
    background: #080d18;
    color: #49576b;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    width: 100%;
    transition: all 0.3s;
}

.stButton > button:hover {
    background: #191f2e !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3) !important;
}

/* Streamlit also wraps text in a <p> inside the button */
.stButton > button:hover p {
    color: #ffffff !important;
}

/* And sometimes in a span */
.stButton > button:hover span {
    color: #ffffff !important;
}
        
        /* Text Colors */
        h1, h2, h3 { color: #f1f5f9; }
        p { color: #94a3b8; }
        
        /* Table styling */
        [data-testid="stDataFrame"] {
            background-color: #1e293b;
        }
/* Prevent nav from collapsing below readable width */
.nav-bar {
    min-width: 600px;
    overflow-x: auto;
}

/* Remove Streamlit's default column padding on nav */
.nav-bar [data-testid="column"] {
    padding: 0 4px !important;
}
            

    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    /* Force all nav buttons to same size, no wrapping */
    .nav-bar .stButton > button {
        white-space: nowrap !important;
        min-width: 100px !important;
        width: 100% !important;
        height: 40px !important;
        padding: 0 16px !important;
        font-size: 14px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

</style>
""", unsafe_allow_html=True)


if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'selected_status' not in st.session_state:
    st.session_state.selected_status = None

def process_csv_via_api(uploaded_file):
    """Sends CSV to FastAPI and gets Real Predictions"""
    try:
       
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0) 
        
       
        files = {'file': (uploaded_file.name, file_bytes, 'text/csv')}
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"❌ API Error: {error_detail}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to Backend. Is FastAPI running on Port 8000?")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        return None

# Navigation Bar 

st.markdown('<div class="nav-bar">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([2.5, 1, 1, 1, 1])

with col1:
    st.markdown("### ⚡ Equip-GuardianAngel")
with col2:
    if st.button("🏠 Home", use_container_width=True): 
        st.session_state.current_page = 'home'; st.rerun()
with col3:
    if st.button("📊 Dashboard", use_container_width=True): 
        st.session_state.current_page = 'dashboard'; st.rerun()
with col4:
    if st.button("📤 Upload", use_container_width=True): 
        st.session_state.current_page = 'upload'; st.rerun()
with col5:
    if st.button("📖 Docs", use_container_width=True): 
        st.session_state.current_page = 'docs'; st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.divider()

# PAGE: HOME 

if st.session_state.current_page == 'home':
    
    st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <p style="font-size: 16px; color: #10a9e8; font-weight: bold; margin-bottom: 10px;">
                AI-POWERED RELIABILITY
            </p>
            <h1 style="font-size: 56px; font-weight: 800; margin-bottom: 20px; line-height: 1.1;">
                Predict Failures <span style="background: -webkit-linear-gradient(left, #06b6d4, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Before Impact</span>
            </h1>
            <p style="font-size: 18px; max-width: 700px; margin: 0 auto 40px;">
                Upload your raw sensor data and get instant Remaining Useful Life (RUL) estimates powered by Deep Learning.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Analysis Now", use_container_width=True):
            st.session_state.current_page = 'upload'
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Features Grid
    st.markdown("<h3 style='text-align: center;'>Platform Capabilities</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
    <div style="background: #0d1626; padding: 25px; border-radius: 12px; 
                border: 1px solid #1e3a5f; height: 100%;">
        <div style="background: #0e3d4a; width: 48px; height: 48px; border-radius: 10px;
                    display: flex; align-items: center; justify-content: center; 
                    font-size: 22px; margin-bottom: 18px;">📊</div>
        <h3 style="color: #f1f5f9; font-size: 18px; margin-bottom: 10px;">Real-Time RUL</h3>
        <p style="color: #64748b;">Precise cycle estimations using RF model trained on CMAPSS data.</p>
    </div>
    """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
    <div style="background: #0d1626; padding: 25px; border-radius: 12px; 
                border: 1px solid #1e3a5f; height: 100%;">
        <div style="background: #0e3d4a; width: 48px; height: 48px; border-radius: 10px;
                    display: flex; align-items: center; justify-content: center; 
                    font-size: 22px; margin-bottom: 18px;">🛡️</div>
        <h3 style="color: #f1f5f9; font-size: 18px; margin-bottom: 10px;">Risk Classification</h3>
        <p style="color: #64748b;">Automatic categorization into Healthy, Warning, and Critical states for prioritization.</p>
    </div>
    """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div style="background: #0d1626;padding:25px; border-radius:12px;
                    border: 1px solid #1e3a5f; height: 100%;">
<div style="background: #0e3d4a; width: 48px;height: 48px; border-radius: 10px;
                    display: flex; align-items: center; justify-content: center; 
                    font-size: 22px; margin-bottom: 18px;">📉</div>
                    <h3 style="color: #f1f5f9; font-size: 18px; margin-bottom: 10px;">Trend Analysis</h3>
        <p style="color: #64748b;">Visualize degradation patterns across your entire equipment fleet instantly.</p>
    </div>
    """, unsafe_allow_html=True)

#PAGE: UPLOAD

elif st.session_state.current_page == 'upload':
    st.markdown("## 📤 Upload Sensor Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: #1e293b; padding: 30px; border-radius: 10px; border: 1px dashed #475569;">
            <p style="margin-bottom: 10px; color: white;"><b>Select CSV File</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Drop CSV here", type=['csv'], label_visibility="collapsed")
        
        if uploaded_file:
            st.info(f"📄 Selected: {uploaded_file.name}")
            
            if st.button("⚡ Run Predictive Model", use_container_width=True):
                with st.spinner("Transmitting data to Neural Network..."):
                    result = process_csv_via_api(uploaded_file)
                    
                    if result:
                        st.session_state.predictions = result # Store API Response
                        st.session_state.selected_status = None
                        st.session_state.current_page = 'dashboard'
                        st.rerun()

    with col2:
        st.markdown("### Requirements")
        st.info("""
        **Format:** CSV or TXT
        
        **Columns Required:**
        - 'id','cycle' (Equipment Identifier)
        - 'setting1', 'setting2', 'sensor_2', 'sensor_3', 'sensor_4', 
        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'(Sensor Readings)
        """)
        st.markdown("Your data is processed securely and is never stored permanently on our servers.")

#PAGE: DASHBOARD

elif st.session_state.current_page == 'dashboard':

    if not st.session_state.predictions:
        st.warning("⚠️ No data loaded. Please upload a file first.")
        if st.button("Go to Upload"):
            st.session_state.current_page = 'upload'
            st.rerun()

    else:
       
        api_data = st.session_state.predictions
        df_pred = pd.DataFrame(api_data['equipment'])

        
        def metric_card(icon, title, count, subtitle, border_color, number_color, key):
            st.markdown(f"""
                <div style="
                    background: #0d1117;
                    border: 1.5px solid {border_color};
                    border-radius: 12px;
                    padding: 24px 28px;
                ">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                        <span style="font-size: 18px;">{icon}</span>
                        <span style="color: #94a3b8; font-size: 15px; font-weight: 500;">{title}</span>
                    </div>
                    <div style="font-size: 52px; font-weight: 700; color: {number_color};
                                line-height: 1; margin-bottom: 14px;">{count}</div>
                    <div style="color: #64748b; font-size: 14px;">{subtitle}</div>
                </div>
            """, unsafe_allow_html=True)
            # hidden button
            st.markdown('<div style="height:0; overflow:hidden;">', unsafe_allow_html=True)
            clicked = st.button(title, key=key, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return clicked

        st.markdown("## 📊 Fleet Status Overview")
        c1, c2, c3 = st.columns(3)

        with c1:
            if metric_card("✅", "Healthy Equipment", api_data['healthy_count'],
                           "Operating normally", "#10b981", "#10b981", "btn_healthy"):
                st.session_state.selected_status = 'Healthy'

        with c2:
            if metric_card("⚠️", "Warning Status", api_data['warning_count'],
                           "Attention needed", "#f59e0b", "#f59e0b", "btn_warning"):
                st.session_state.selected_status = 'Warning'

        with c3:
            if metric_card("🔴", "Critical Alerts", api_data['critical_count'],
                           "Immediate action required", "#ef4444", "#ef4444", "btn_critical"):
                st.session_state.selected_status = 'Critical'


        st.markdown("---")

        
        filtered_df = df_pred.copy()
        if st.session_state.selected_status:
            filtered_df = df_pred[df_pred['status'] == st.session_state.selected_status]
            st.markdown(f"### 🔍 Showing: {st.session_state.selected_status} Equipment")
            if st.button("Clear Filter", key="clr_filter"):
                st.session_state.selected_status = None
                st.rerun()

       
        if not filtered_df.empty:

            fig = px.bar(
                filtered_df.sort_values('cycles', ascending=True),
                x='cycles',
                y='id',
                orientation='h',
                color='status',
                color_discrete_map={
                    'Healthy': '#10b981',
                    'Warning': '#f59e0b',
                    'Critical': '#ef4444'
                },
                title="Estimated Remaining Cycles per Unit"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                yaxis=dict(type='category')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Data Table
            st.markdown("### 📋 Detailed Data Registry")
            table_df = filtered_df[['id', 'status', 'cycles']].copy()
            st.dataframe(
                table_df,
                use_container_width=True,
                column_config={
                    "id": "Equipment ID",
                    "status": "Health Status",
                    "cycles": st.column_config.ProgressColumn(
                        "Remaining Cycles",
                        help="Predicted remaining useful life",
                        format="%d",
                        min_value=0,
                        max_value=125,
                    )
                },
                hide_index=True
            )

            # Download
            csv = table_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Report",
                csv,
                f"maintenance_report_{datetime.now().strftime('%H%M%S')}.csv",
                "text/csv",
                key='download-csv'
            )

        else:
            st.info("No equipment found in this category.")

# PAGE: DOCS

elif st.session_state.current_page == 'docs':
    st.markdown("## 📖 Documentation")
    st.info("This interface is connected to a FastAPI backend running a RF Model Trained on CMAPSS.")
    
    st.markdown("""
    ### Prediction Logic
    1. **Data Ingestion**: CSV uploaded via frontend.
    2. **API Transmission**: Data sent to `/api/v1/predict`.
    3. **Preprocessing**: Backend uses saved `StandardScaler` to normalize inputs.
    4. **Inference**: Loaded Model predicts RUL (0-125 scale).
    5. **Thresholding**:
        - **Critical**: < 30 Cycles
        - **Warning**: 30 - 50 Cycles
        - **Healthy**: > 50 Cycles
    """)
