import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline, interp1d
import pickle
import io

# --- Configuration & Styling ---
st.set_page_config(
    page_title="DSC Peak Integrator",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
    }
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #95a5a6;
    }
    .lock-status {
        font-weight: bold;
        padding: 5px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 10px;
    }
    .locked {
        background-color: #ffcccc;
        color: #990000;
        border: 1px solid #990000;
    }
    .unlocked {
        background-color: #e6fffa;
        color: #006666;
        border: 1px solid #006666;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Management ---
def init_session_state():
    """Initialize session state variables to persist settings."""
    # Global Defaults
    if "global_settings" not in st.session_state:
        st.session_state.global_settings = {
            "peak_range": None,    # (start, end)
            "fit_regions": None,   # [(start, end), (start, end)]
            "method": "Sigmoidal (Tm-based)",
            "poly_degree": 3,
            "sigmoid_width": 2.0,
            "smooth_factor": 0.0,
            "tm_manual": None
        }
    
    # Storage for Locked Files: {filename: {settings_dict}}
    if "locked_configs" not in st.session_state:
        st.session_state.locked_configs = {}

    # Storage for Loaded DataFrames (for Session Persistence)
    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = {}
        
    # Storage for Renamed Files: {original_filename: display_name}
    if "file_renames" not in st.session_state:
        st.session_state.file_renames = {}

init_session_state()

# --- Helper Functions ---

def generate_mock_data():
    """Generates a synthetic DSC thermogram with noise and baseline drift."""
    T = np.linspace(20, 110, 500)
    
    # Sigmoidal baseline shift (Heat Capacity change upon unfolding)
    base_native = 0.5 + 0.002 * T 
    base_unfolded = 1.2 + 0.0025 * T
    
    # Transition progress (Logistic function)
    Tm = 65.0
    width = 5.0
    alpha = 1 / (1 + np.exp(-(T - Tm) / width))
    
    # Combined baseline
    baseline = (1 - alpha) * base_native + alpha * base_unfolded
    
    # Peak (Gaussian approx for transition enthalpy)
    peak_height = 2.5
    peak = peak_height * np.exp(-0.5 * ((T - Tm) / (width * 0.8))**2)
    
    # Add some random noise and instrument curvature
    noise = np.random.normal(0, 0.02, len(T))
    curvature = 0.00005 * (T - 60)**2
    
    Cp = baseline + peak + noise + curvature
    
    return pd.DataFrame({"Temperature": T, "Cp": Cp})

def clean_and_standardize_data(df, target_temp_col=None, target_cp_col=None, ref_grid=None):
    """Cleans data, resolves column mismatches, and optionally interpolates."""
    if target_temp_col in df.columns and target_cp_col in df.columns:
        temp_data = df[target_temp_col]
        cp_data = df[target_cp_col]
    elif len(df.columns) >= 2:
        temp_data = df.iloc[:, 0]
        cp_data = df.iloc[:, 1]
    else:
        return None

    clean_df = pd.DataFrame({"Temperature": temp_data, "Cp": cp_data})
    clean_df["Temperature"] = pd.to_numeric(clean_df["Temperature"], errors='coerce')
    clean_df["Cp"] = pd.to_numeric(clean_df["Cp"], errors='coerce')
    clean_df = clean_df.dropna().sort_values("Temperature")
    clean_df = clean_df.drop_duplicates(subset=["Temperature"])
    
    if clean_df.empty:
        return None

    if ref_grid is not None:
        try:
            f = interp1d(clean_df["Temperature"], clean_df["Cp"], 
                         kind='linear', bounds_error=False, fill_value=np.nan)
            new_cp = f(ref_grid)
            clean_df = pd.DataFrame({"Temperature": ref_grid, "Cp": new_cp}).dropna()
        except Exception:
            return None

    return clean_df

def fit_baseline(df, method, fit_regions, smooth_factor=None, poly_order=3, tm_manual=None, sigmoid_width=2.0):
    """Constructs a baseline based on selected regions."""
    mask = np.zeros(len(df), dtype=bool)
    for start, end in fit_regions:
        mask |= (df['Temperature'] >= start) & (df['Temperature'] <= end)
    
    X_fit = df.loc[mask, 'Temperature'].values
    Y_fit = df.loc[mask, 'Cp'].values
    X_all = df['Temperature'].values
    
    if len(X_fit) < 2:
        return np.zeros_like(X_all)

    if method == "Linear Connect":
        if len(fit_regions) >= 2:
            left_mask = (df['Temperature'] >= fit_regions[0][0]) & (df['Temperature'] <= fit_regions[0][1])
            right_mask = (df['Temperature'] >= fit_regions[-1][0]) & (df['Temperature'] <= fit_regions[-1][1])
            
            p1 = (df.loc[left_mask, 'Temperature'].mean(), df.loc[left_mask, 'Cp'].mean())
            p2 = (df.loc[right_mask, 'Temperature'].mean(), df.loc[right_mask, 'Cp'].mean())
            
            if np.isnan(p1[0]) or np.isnan(p2[0]): return np.zeros_like(X_all)
            
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            c = p1[1] - m * p1[0]
            return m * X_all + c
        else:
            z = np.polyfit(X_fit, Y_fit, 1)
            return np.poly1d(z)(X_all)

    elif method == "Sigmoidal (Tm-based)":
        mask_left = (df['Temperature'] >= fit_regions[0][0]) & (df['Temperature'] <= fit_regions[0][1])
        if mask_left.sum() > 1:
            z_left = np.polyfit(df.loc[mask_left, 'Temperature'], df.loc[mask_left, 'Cp'], 1)
            base_folded = np.poly1d(z_left)(X_all)
        else:
            base_folded = np.full_like(X_all, df.loc[mask_left, 'Cp'].mean() if mask_left.any() else 0)

        mask_right = (df['Temperature'] >= fit_regions[-1][0]) & (df['Temperature'] <= fit_regions[-1][1])
        if mask_right.sum() > 1:
            z_right = np.polyfit(df.loc[mask_right, 'Temperature'], df.loc[mask_right, 'Cp'], 1)
            base_unfolded = np.poly1d(z_right)(X_all)
        else:
            base_unfolded = np.full_like(X_all, df.loc[mask_right, 'Cp'].mean() if mask_right.any() else 0)
        
        if tm_manual is None:
            gap_mask = (df['Temperature'] > fit_regions[0][1]) & (df['Temperature'] < fit_regions[-1][0])
            if gap_mask.any():
                idx_max = df.loc[gap_mask, 'Cp'].argmax()
                tm_val = df.loc[gap_mask].iloc[idx_max]['Temperature']
            else:
                tm_val = df['Temperature'].mean()
        else:
            tm_val = tm_manual

        w = sigmoid_width if sigmoid_width > 0.1 else 0.1
        alpha = 1.0 / (1.0 + np.exp(-(X_all - tm_val) / w))
        
        return (1 - alpha) * base_folded + alpha * base_unfolded

    elif method == "Polynomial":
        try:
            z = np.polyfit(X_fit, Y_fit, poly_order)
            return np.poly1d(z)(X_all)
        except Exception:
            return np.zeros_like(X_all)

    elif method == "Cubic Spline":
        sort_idx = np.argsort(X_fit)
        X_fit_sorted = X_fit[sort_idx]
        Y_fit_sorted = Y_fit[sort_idx]
        try:
            spl = UnivariateSpline(X_fit_sorted, Y_fit_sorted, k=3, s=smooth_factor)
            return spl(X_all)
        except Exception:
            return np.zeros_like(X_all)
            
    return np.zeros_like(X_all)

def calculate_metrics(df, peak_start, peak_end, unit="kcal"):
    """
    Calculates thermodynamic metrics using the Integral Width method.
    
    Args:
        df: DataFrame with 'Temperature' and 'Delta_Cp'
        peak_start, peak_end: Integration limits
        unit: 'kcal', 'cal' or 'J' for determining Gas Constant R
        
    Returns:
        dict: containing all metrics
    """
    # Gas Constant R
    # kcal/mol/K = 1.9872e-3
    # cal/mol/K = 1.9872
    # J/mol/K = 8.314
    
    if unit == "kcal":
        R = 1.9872e-3  # If inputs are kcal, R must be kcal
    elif unit == "cal":
        R = 1.9872
    else: # J
        R = 8.314
    
    mask_peak = (df['Temperature'] >= peak_start) & (df['Temperature'] <= peak_end)
    df_peak = df[mask_peak]
    
    # Initialize with all keys to prevent KeyErrors
    metrics = {
        "area": 0.0, 
        "tm": 0.0, 
        "cp_max": 0.0,
        "width_int": 0.0, 
        "dh_vh": 0.0,
        "coop_ratio": 0.0
    }
    
    if not df_peak.empty:
        # 1. Calorimetric Enthalpy (Area) - Model Free
        area = simpson(df_peak['Delta_Cp'], x=df_peak['Temperature'])
        metrics["area"] = area
        
        # 2. Tm and Cp_max (Peak Height)
        # Find index of max value
        peak_max_idx = df_peak['Delta_Cp'].argmax()
        tm = df_peak.iloc[peak_max_idx]['Temperature']
        cp_max = df_peak.iloc[peak_max_idx]['Delta_Cp']
        metrics["tm"] = tm
        metrics["cp_max"] = cp_max
        
        tm_k = tm + 273.15
        
        if cp_max > 1e-9:
            # 3. Integral Width
            # dT = Area / Height
            width_int = area / cp_max
            metrics["width_int"] = width_int
            
            # 4. Van't Hoff Enthalpy
            # Formula: dH_vh = 4 * R * Tm^2 * Cp_max / dH_cal
            # Equivalent to: 4 * R * Tm^2 / width_int
            if abs(area) > 1e-6:
                dh_vh = (4 * R * (tm_k**2) * cp_max) / area
                metrics["dh_vh"] = dh_vh
                metrics["coop_ratio"] = area / dh_vh

    return metrics

def load_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception:
        return None

def clamp(val, min_v, max_v):
    """Helper to ensure value stays within bounds."""
    if val is None: return None
    return max(min_v, min(val, max_v))

# --- Plotting Helpers (Axes & Style) ---
def hex_to_rgba(hex_color, opacity=0.2):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{opacity})"
    return f"rgba(0,0,0,{opacity})"

def update_axes_style(fig, x_label, y_label, config):
    """Applies the boxed look (mirror axes), thickness, and labels."""
    font_family = config.get("font_family", "Arial")
    title_font_size = config.get("title_font_size", 16)
    title_color = config.get("title_color", "black")
    tick_font_size = config.get("tick_font_size", 14)
    label_color = config.get("label_color", "black")

    axis_line_width = config.get("axis_width", 2)
    axis_line_color = config.get("axis_line_color", "black")
    show_grid = config.get("show_grid", False)
    grid_color = config.get("grid_color", "#e5e7eb")
    
    major_tick_len = config.get("major_tick_len", 6)
    major_tick_width = config.get("major_tick_width", 2)
    major_tick_color = config.get("major_tick_color", "black")
    
    show_minor = config.get("show_minor_ticks", False)
    minor_tick_len = config.get("minor_tick_len", 3)
    minor_tick_width = config.get("minor_tick_width", 1)
    minor_tick_color = config.get("minor_tick_color", "black")
    
    legend_pos = config.get("legend_pos", "Outside Right")
    show_leg_border = config.get("show_legend_border", True)
    legend_bg_color = config.get("legend_bg_color", "rgba(255,255,255,0.8)")
    
    # Legend Configuration
    legend_dict = dict(
        font=dict(size=tick_font_size),
        bgcolor=legend_bg_color, 
        bordercolor="black" if show_leg_border else "rgba(0,0,0,0)",
        borderwidth=1 if show_leg_border else 0
    )
    
    if legend_pos == "Inside Top-Right":
        legend_dict.update(dict(yanchor="top", y=0.98, xanchor="right", x=0.98))
    elif legend_pos == "Inside Top-Left":
        legend_dict.update(dict(yanchor="top", y=0.98, xanchor="left", x=0.02))
    elif legend_pos == "Inside Bottom-Right":
        legend_dict.update(dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98))
    elif legend_pos == "Inside Bottom-Left":
        legend_dict.update(dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02))
    else: 
        legend_dict.update(dict(yanchor="top", y=1, xanchor="left", x=1.02))

    fig.update_layout(
        template="simple_white",
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=config.get("show_legend", True),
        legend=legend_dict,
        font=dict(family=font_family, color="black"), 
        hovermode="x unified",
        margin=dict(l=80, r=40, t=50, b=80),
        paper_bgcolor=config.get("bg_color", "#FFFFFF"),
        plot_bgcolor=config.get("bg_color", "#FFFFFF"),
    )
    
    major_dtick_x = config.get("x_dtick", 0)
    major_dtick_y = config.get("y_dtick", 0)
    minor_subdivisions = config.get("minor_subdivisions", 5) 

    minor_dict = dict(
        ticklen=minor_tick_len,
        tickwidth=minor_tick_width,
        tickcolor=minor_tick_color,
        ticks="outside",
        showgrid=False
    )

    base_axis_dict = dict(
        showline=True,
        linewidth=axis_line_width,
        linecolor=axis_line_color,
        mirror=True,
        ticks="outside",
        tickwidth=major_tick_width,
        ticklen=major_tick_len,
        tickcolor=major_tick_color,
        title_font=dict(size=title_font_size, color=title_color),
        tickfont=dict(size=tick_font_size, color=label_color),
        showgrid=show_grid,
        gridcolor=grid_color,
    )

    x_tick_dict = base_axis_dict.copy()
    if major_dtick_x > 0:
        x_tick_dict["dtick"] = major_dtick_x
        if show_minor:
            m_dict = minor_dict.copy()
            m_dict["dtick"] = major_dtick_x / minor_subdivisions
            x_tick_dict["minor"] = m_dict
    else:
        if show_minor:
             x_tick_dict["minor"] = minor_dict.copy()
            
    y_tick_dict = base_axis_dict.copy()
    if major_dtick_y > 0:
        y_tick_dict["dtick"] = major_dtick_y
        if show_minor:
            m_dict = minor_dict.copy()
            m_dict["dtick"] = major_dtick_y / minor_subdivisions
            y_tick_dict["minor"] = m_dict
    else:
        if show_minor:
             y_tick_dict["minor"] = minor_dict.copy()

    fig.update_xaxes(**x_tick_dict)
    fig.update_yaxes(**y_tick_dict)
    
    # Custom Ranges (Trimming)
    x_range = config.get("x_range")
    y_range = config.get("y_range")
    if x_range and len(x_range) == 2 and x_range[0] < x_range[1]:
        fig.update_xaxes(range=x_range)
    if y_range and len(y_range) == 2 and y_range[0] < y_range[1]:
        fig.update_yaxes(range=y_range)
    
    fig.update_layout(width=config.get("width", 800), height=config.get("height", 600))
    return fig

# --- Main Application ---

st.markdown('<div class="main-header">DSC Thermogram Analysis</div>', unsafe_allow_html=True)

# --- Sidebar: Tabs for Data and Style ---
sidebar_tab_data, sidebar_tab_style = st.sidebar.tabs(["Data & Analysis", "Style & Axes"])

# =========================================================
# TAB: Data & Analysis (Original Functionality)
# =========================================================
with sidebar_tab_data:
    st.header("1. Session & Data")
    
    # Session I/O
    with st.expander("💾 Save / Load Session"):
        # We need to capture ALL session state keys related to style as well.
        style_keys = [
            "fig_w", "fig_h", "trans_bg", "bg_col", "curve_wid", "axis_thick", "axis_col",
            "show_grid", "grid_col", "font_fam", "title_font_sz", "title_col", "tick_font_sz", "label_col",
            "maj_tick_len", "maj_tick_wid", "maj_tick_col", "x_dtick", "y_dtick", "show_minor",
            "min_tick_len", "min_tick_wid", "min_tick_col", "minor_divs", "show_leg", "show_leg_border",
            "trans_leg_bg", "leg_bg_col", "leg_bg_opacity", "leg_pos_sel", "cx_check", "cx_min", "cx_max",
            "cy_check", "cy_min", "cy_max", "custom_x_title", "custom_y_title"
        ]
        style_state = {k: st.session_state[k] for k in style_keys if k in st.session_state}

        session_data = {
            "loaded_data": st.session_state.loaded_data,
            "locked_configs": st.session_state.locked_configs,
            "global_settings": st.session_state.global_settings,
            "file_renames": st.session_state.file_renames,
            "style_settings": style_state
        }
        buffer = io.BytesIO()
        pickle.dump(session_data, buffer)
        buffer.seek(0)
        
        st.download_button(
            label="Download Current Session (.dsc)",
            data=buffer,
            file_name="analysis_session.dsc",
            mime="application/octet-stream"
        )
        
        # LOAD
        uploaded_session = st.file_uploader("Load Session File", type=[".dsc"])
        if uploaded_session:
            try:
                loaded_state = pickle.load(uploaded_session)
                st.session_state.loaded_data = loaded_state.get("loaded_data", {})
                st.session_state.locked_configs = loaded_state.get("locked_configs", {})
                st.session_state.global_settings = loaded_state.get("global_settings", {})
                st.session_state.file_renames = loaded_state.get("file_renames", {})
                
                # Restore style settings
                saved_styles = loaded_state.get("style_settings", {})
                for k, v in saved_styles.items():
                    st.session_state[k] = v
                    
                st.success("Session loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load session: {e}")

    # Data Input
    mode = st.radio("Input Mode", ["File Upload / Batch", "Use Mock Data"])
    
    # Energy Unit Selector
    st.markdown("**Parameters**")
    energy_unit = st.radio("Input Energy Units (defines R)", ["kcal", "cal", "J"], horizontal=True, 
                           help="Commonly kcal. R will be set accordingly: 1.987e-3 (kcal), 1.987 (cal), or 8.314 (J)")

    if mode == "File Upload / Batch":
        uploaded_files = st.file_uploader("Add Files", type=["csv", "xlsx"], accept_multiple_files=True)
        if uploaded_files:
            for uf in uploaded_files:
                if uf.name not in st.session_state.loaded_data:
                    temp_df = load_file(uf)
                    if temp_df is not None:
                        st.session_state.loaded_data[uf.name] = temp_df
        
        # --- File Management (Unload) ---
        if st.session_state.loaded_data:
            with st.expander("Manage Loaded Files"):
                files_to_unload = st.multiselect("Select files to unload", list(st.session_state.loaded_data.keys()))
                if st.button("Unload Selected Files"):
                    for fname in files_to_unload:
                        del st.session_state.loaded_data[fname]
                        # Clean up associated configs
                        if fname in st.session_state.locked_configs:
                            del st.session_state.locked_configs[fname]
                        if fname in st.session_state.file_renames:
                            del st.session_state.file_renames[fname]
                    st.rerun()
                
                if st.button("Clear All Data", type="primary"):
                    st.session_state.loaded_data = {}
                    st.session_state.locked_configs = {}
                    st.session_state.file_renames = {}
                    st.rerun()

    else:
        if "Mock Data" not in st.session_state.loaded_data:
            st.session_state.loaded_data["Mock Data"] = generate_mock_data()

    # --- Active File Selection & Renaming ---
    df_active = None
    active_filename = None
    lock_toggle = False 

    if st.session_state.loaded_data:
        st.markdown("---")
        file_options = list(st.session_state.loaded_data.keys())
        def get_display_name(fname): return st.session_state.file_renames.get(fname, fname)
        
        active_filename = st.selectbox("Select File to View/Edit", file_options, format_func=get_display_name)
        
        current_display = get_display_name(active_filename)
        new_name = st.text_input("Rename Plot/File", value=current_display)
        if new_name != current_display:
            st.session_state.file_renames[active_filename] = new_name
            st.rerun()
        
        df_raw = st.session_state.loaded_data[active_filename]
        cols = df_raw.columns.tolist()
        st.subheader("Reference Column Mapping")
        temp_col = st.selectbox("Temperature Column", cols, index=0 if len(cols)>0 else -1)
        cp_col = st.selectbox("Signal Column (Cp)", cols, index=1 if len(cols)>1 else -1)
        homogenize_data = st.checkbox("Homogenize Data (Interpolate)", value=True, help="Standardizes temperature grid across files for comparison.")
        
        is_locked_state = active_filename in st.session_state.locked_configs
        lock_toggle = st.checkbox(
            "🔒 Lock Settings for this File", 
            value=is_locked_state, 
            key=f"lock_{active_filename}",
            help="If checked, this file's settings are saved and won't change when you adjust global settings."
        )
        
        if temp_col and cp_col:
            df_active = clean_and_standardize_data(df_raw, temp_col, cp_col)
            if df_active is not None and homogenize_data:
                 master_grid = df_active['Temperature'].values 
            else:
                 master_grid = None
        else:
            st.error("Select valid columns.")

# =========================================================
# TAB: Style & Axes (New Aesthetic Controls)
# =========================================================
with sidebar_tab_style:
    st.header("Plot Aesthetics")
    
    with st.expander("Axis Labels", expanded=True):
        custom_x_title = st.text_input("X-Axis Title", value="Temperature (°C)", key="custom_x_title")
        custom_y_title = st.text_input("Y-Axis Title", value="Cp (mcal/mol/K)", key="custom_y_title")

    with st.expander("Layout & Dimensions", expanded=False):
        c_fig1, c_fig2 = st.columns(2)
        canvas_width = c_fig1.number_input("Width (px)", 400, 4000, 800, 50, key="fig_w")
        canvas_height = c_fig2.number_input("Height (px)", 300, 4000, 550, 50, key="fig_h")
    
    with st.expander("Background Styling", expanded=False):
        transparent_bg = st.checkbox("Transparent Background", value=False, key="trans_bg")
        if not transparent_bg:
            bg_col_val = st.color_picker("Background Color", "#FFFFFF", key="bg_col")
        else:
            bg_col_val = "rgba(0,0,0,0)"

    with st.expander("Curve Styling", expanded=False):
        curve_width = st.slider("Curve Line Width", 1, 10, 2, key="curve_wid")
    
    with st.expander("Axis Lines & Grids"):
        c_al1, c_al2 = st.columns(2)
        axis_width = c_al1.slider("Line Thickness", 1, 5, 2, key="axis_thick")
        axis_line_color = c_al2.color_picker("Line Color", "#000000", key="axis_col")
        
        c_gl1, c_gl2 = st.columns(2)
        show_grid = c_gl1.checkbox("Show Grid Lines", False, key="show_grid")
        grid_color = c_gl2.color_picker("Grid Color", "#e5e7eb", key="grid_col")

    with st.expander("Typography (Fonts)"):
        available_fonts = ["Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New", "Georgia", "Trebuchet MS", "Impact"]
        font_family = st.selectbox("Font Family", available_fonts, index=0, key="font_fam")
        
        st.markdown("**Axis Titles**")
        c_tf1, c_tf2 = st.columns(2)
        title_font_size = c_tf1.number_input("Size", 8, 48, 16, key="title_font_sz")
        title_color = c_tf2.color_picker("Color", "#000000", key="title_col")
        
        st.markdown("**Tick Labels**")
        c_lf1, c_lf2 = st.columns(2)
        tick_font_size = c_lf1.number_input("Size", 8, 36, 14, key="tick_font_sz")
        label_color = c_lf2.color_picker("Color", "#000000", key="label_col")

    with st.expander("Ticks & Intervals"):
        st.markdown("**Major Ticks**")
        c_maj1, c_maj2, c_maj3 = st.columns(3)
        major_tick_len = c_maj1.number_input("Length", 0, 20, 6, key="maj_tick_len")
        major_tick_width = c_maj2.number_input("Width", 1, 5, 2, key="maj_tick_wid")
        major_tick_color = c_maj3.color_picker("Color", "#000000", key="maj_tick_col")
        
        c_gap1, c_gap2 = st.columns(2)
        x_dtick = c_gap1.number_input("X Step (0=Auto)", 0.0, 500.0, 0.0, 5.0, key="x_dtick")
        y_dtick = c_gap2.number_input("Y Step (0=Auto)", 0.0, 100.0, 0.0, 1.0, key="y_dtick")

        st.markdown("---")
        st.markdown("**Minor Ticks**")
        show_minor_ticks = st.checkbox("Show Minor Ticks", False, key="show_minor")
        
        c_min1, c_min2, c_min3 = st.columns(3)
        minor_tick_len = c_min1.number_input("Length", 0, 10, 3, key="min_tick_len")
        minor_tick_width = c_min2.number_input("Width", 1, 5, 1, key="min_tick_wid")
        minor_tick_color = c_min3.color_picker("Color", "#000000", key="min_tick_col")

        minor_subdivisions = st.number_input("Subdivisions (Intervals)", 1, 20, 5, key="minor_divs")

    with st.expander("Legend"):
        show_legend = st.checkbox("Show Legends", True, key="show_leg")
        show_legend_border = st.checkbox("Show Legend Border", True, key="show_leg_border")
        
        transparent_leg_bg = st.checkbox("Transparent Legend Background", False, key="trans_leg_bg")
        if not transparent_leg_bg:
            leg_bg_col_val = st.color_picker("Legend Background Color", "#FFFFFF", key="leg_bg_col")
            leg_bg_opacity = st.slider("Legend Background Opacity", 0.0, 1.0, 0.8, 0.1, key="leg_bg_opacity")
            legend_bg_color = hex_to_rgba(leg_bg_col_val, leg_bg_opacity)
        else:
            legend_bg_color = "rgba(0,0,0,0)"

        legend_options = ["Outside Right", "Inside Top-Right", "Inside Top-Left", "Inside Bottom-Right", "Inside Bottom-Left"]
        legend_pos = st.selectbox("Position", legend_options, index=0, key="leg_pos_sel")

    st.markdown("---")
    with st.expander("Global Plot Ranges (Trim)", expanded=True):
        st.caption("Custom Limits (Leave unchecked for auto)")
        
        # X-axis
        custom_x = st.checkbox("Custom X-Axis Range", key="cx_check")
        x_range = None
        if custom_x:
            c1, c2 = st.columns(2)
            xm = c1.number_input("Min X", value=20.0, key="cx_min")
            xM = c2.number_input("Max X", value=120.0, key="cx_max")
            if xm < xM: x_range = [xm, xM]

        # Y-axis 
        custom_y = st.checkbox("Custom Y-Axis Range", key="cy_check")
        y_range = None
        if custom_y:
            c1, c2 = st.columns(2)
            ym = c1.number_input("Min Y", value=0.0, key="cy_min")
            yM = c2.number_input("Max Y", value=5.0, key="cy_max")
            if ym < yM: y_range = [ym, yM]


# --- Assemble Style Config ---
style_config = {
    "width": canvas_width, "height": canvas_height,
    "x_range": x_range, "y_range": y_range,
    "x_label": custom_x_title, "y_label": custom_y_title, # New custom labels
    "show_legend": show_legend, 
    "legend_pos": legend_pos,
    "axis_width": axis_width, "axis_line_color": axis_line_color,
    "show_grid": show_grid, "grid_color": grid_color,
    "font_family": font_family,
    "title_font_size": title_font_size, "title_color": title_color,
    "tick_font_size": tick_font_size, "label_color": label_color,
    "major_tick_len": major_tick_len, "major_tick_width": major_tick_width, "major_tick_color": major_tick_color,
    "show_minor_ticks": show_minor_ticks,
    "minor_tick_len": minor_tick_len, "minor_tick_width": minor_tick_width, "minor_tick_color": minor_tick_color,
    "minor_subdivisions": minor_subdivisions,
    "x_dtick": x_dtick, "y_dtick": y_dtick,
    "curve_width": curve_width,
    "show_legend_border": show_legend_border,
    "bg_color": bg_col_val,
    "legend_bg_color": legend_bg_color
}


# --- TABS FOR ANALYSIS ---

if df_active is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Single Analysis", "📈 Multi-Plot Comparison", "📋 Batch Results"])

    # ==========================
    # TAB 1: Single Analysis
    # ==========================
    with tab1:
        col_viz, col_controls = st.columns([3, 1])
        
        with col_controls:
            st.header("Settings")
            
            min_t, max_t = float(df_active["Temperature"].min()), float(df_active["Temperature"].max())
            
            # Use the sidebar lock toggle to determine visual status
            if lock_toggle:
                st.markdown(f'<div class="lock-status locked">🔒 Settings Locked</div>', unsafe_allow_html=True)
                current_settings = st.session_state.locked_configs.get(active_filename, st.session_state.global_settings)
            else:
                st.markdown(f'<div class="lock-status unlocked">🔓 Using Global Settings</div>', unsafe_allow_html=True)
                current_settings = st.session_state.global_settings

            # 1. Retrieve & Clamp Peak Range
            stored_range = current_settings.get("peak_range")
            if stored_range:
                val_start = clamp(stored_range[0], min_t, max_t)
                val_end = clamp(stored_range[1], min_t, max_t)
                range_value = (val_start, val_end)
            else:
                range_value = (min_t + (max_t-min_t)*0.3, min_t + (max_t-min_t)*0.7)

            peak_range = st.slider("Integration Range", min_t, max_t, range_value, key="peak_slider")
            peak_start, peak_end = peak_range
            
            # 2. Baseline Method
            baseline_method = st.selectbox(
                "Baseline Type", 
                ["Linear Connect", "Sigmoidal (Tm-based)", "Polynomial", "Cubic Spline"],
                index=["Linear Connect", "Sigmoidal (Tm-based)", "Polynomial", "Cubic Spline"].index(current_settings.get("method", "Sigmoidal (Tm-based)"))
            )
            
            # 3. Method Parameters
            smooth_val = current_settings.get("smooth_factor", 0.0)
            poly_deg = current_settings.get("poly_degree", 3)
            sig_width = current_settings.get("sigmoid_width", 2.0)
            tm_est_manual = current_settings.get("tm_manual", None)
            
            if baseline_method == "Cubic Spline":
                smooth_val = st.number_input("Smoothing (s)", min_value=0.0, step=10.0, value=float(smooth_val))
            elif baseline_method == "Polynomial":
                poly_deg = st.slider("Degree", 1, 5, value=int(poly_deg))
            elif baseline_method == "Sigmoidal (Tm-based)":
                peak_subset = df_active[(df_active['Temperature'] >= peak_start) & (df_active['Temperature'] <= peak_end)]
                est_tm_auto = peak_subset.loc[peak_subset['Cp'].idxmax(), 'Temperature'] if not peak_subset.empty else (peak_start+peak_end)/2
                st.markdown(f"**Est. Tm (Auto):** {est_tm_auto:.2f} °C")
                
                use_manual_tm = st.checkbox("Manual Tm Override?", value=(tm_est_manual is not None))
                if use_manual_tm:
                    default_tm = tm_est_manual if tm_est_manual else est_tm_auto
                    tm_est_manual = st.number_input("Manual Tm", value=float(clamp(default_tm, min_t, max_t)), min_value=min_t, max_value=max_t)
                else:
                    tm_est_manual = None
                    
                sig_width = st.number_input("Sigmoid Width", min_value=0.1, step=0.1, value=float(sig_width))
                
            st.markdown("---")
            st.write("**Baseline Regions**")
            
            stored_regions = current_settings.get("fit_regions")
            if stored_regions:
                fs = clamp(stored_regions[0][0], min_t, max_t)
                ue = clamp(stored_regions[1][1], min_t, max_t)
            else:
                fs = min_t
                ue = max_t

            folded_start = st.number_input("Folded Start", value=float(fs), min_value=min_t, max_value=max_t)
            folded_end = st.number_input("Folded End (Linked to Peak Start)", value=float(peak_start), min_value=min_t, max_value=max_t)
            unfolded_start = st.number_input("Unfolded Start (Linked to Peak End)", value=float(peak_end), min_value=min_t, max_value=max_t)
            unfolded_end = st.number_input("Unfolded End", value=float(ue), min_value=min_t, max_value=max_t)
            
            fit_regions = [(folded_start, folded_end), (unfolded_start, unfolded_end)]

            new_settings = {
                "peak_range": peak_range,
                "fit_regions": fit_regions,
                "method": baseline_method,
                "poly_degree": poly_deg,
                "sigmoid_width": sig_width,
                "smooth_factor": smooth_val,
                "tm_manual": tm_est_manual
            }
            
            if lock_toggle:
                st.session_state.locked_configs[active_filename] = new_settings
            else:
                if active_filename in st.session_state.locked_configs:
                    del st.session_state.locked_configs[active_filename]
                st.session_state.global_settings = new_settings

        # Plotting Single Analysis
        df_active['Baseline'] = fit_baseline(df_active, baseline_method, fit_regions, smooth_val, poly_deg, tm_est_manual, sig_width)
        df_active['Delta_Cp'] = df_active['Cp'] - df_active['Baseline']
        
        # Calculate metrics using robust (integral) method only
        metrics = calculate_metrics(df_active, peak_start, peak_end, energy_unit)

        with col_viz:
            fig = go.Figure()
            
            # Apply Style Settings via Helper
            display_title = get_display_name(active_filename)
            
            fig.add_trace(go.Scatter(x=df_active['Temperature'], y=df_active['Cp'], mode='lines', name='Raw Data', line=dict(color='black', width=style_config['curve_width']), opacity=0.8))
            fig.add_trace(go.Scatter(x=df_active['Temperature'], y=df_active['Baseline'], mode='lines', name=f'Baseline', line=dict(color='red', width=style_config['curve_width'], dash='dash')))
            
            mask_peak = (df_active['Temperature'] >= peak_start) & (df_active['Temperature'] <= peak_end)
            df_fill = df_active[mask_peak]
            if not df_fill.empty:
                fig.add_trace(go.Scatter(
                    x=pd.concat([df_fill['Temperature'], df_fill['Temperature'][::-1]]),
                    y=pd.concat([df_fill['Cp'], df_fill['Baseline'][::-1]]),
                    fill='toself', fillcolor='rgba(0, 100, 250, 0.3)', line=dict(width=0), name='Area'
                ))
                
            for start, end in fit_regions:
                reg_mask = (df_active['Temperature'] >= start) & (df_active['Temperature'] <= end)
                if not reg_mask.empty:
                    fig.add_trace(go.Scatter(x=df_active.loc[reg_mask, 'Temperature'], y=df_active.loc[reg_mask, 'Cp'], mode='lines', line=dict(color='green', width=4), opacity=0.4, showlegend=False))

            fig.add_vline(x=peak_start, line_dash="dot", line_color="gray")
            fig.add_vline(x=peak_end, line_dash="dot", line_color="gray")
            
            fig.update_layout(title=f"Analysis: {display_title}")
            
            # Apply Advanced Styling
            fig = update_axes_style(fig, style_config['x_label'], style_config['y_label'], style_config)
            
            st.plotly_chart(fig, use_container_width=True)

            # Display Stats in a clean layout
            st.markdown("### Thermodynamic Metrics")
            
            c_main1, c_main2, c_main3 = st.columns(3)
            with c_main1:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-label">Calorimetric Enthalpy (ΔH<sub>cal</sub>)</div>
                    <div class="metric-value">{:.2e}</div>
                    <div class="metric-sub">Area under curve</div>
                </div>
                """.format(metrics['area']), unsafe_allow_html=True)
            
            with c_main2:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-label">Melting Temp (T<sub>m</sub>)</div>
                    <div class="metric-value">{:.2f} °C</div>
                    <div class="metric-sub">Peak maximum</div>
                </div>
                """.format(metrics['tm']), unsafe_allow_html=True)
            
            with c_main3:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-label">Cooperativity (ΔH<sub>cal</sub> / ΔH<sub>vH</sub>)</div>
                    <div class="metric-value">{:.2f}</div>
                    <div class="metric-sub">Ratio (n)</div>
                </div>
                """.format(metrics['coop_ratio']), unsafe_allow_html=True)

            st.markdown("#### Detailed Analysis & Intermediate Values")
            c_det1, c_det2 = st.columns(2)
            
            with c_det1:
                st.markdown("**Peak Data**")
                st.write(f"• Height ($C_{{p,max}}$): `{metrics['cp_max']:.3f}`")
                st.write(f"• Area ($\Delta H_{{cal}}$): `{metrics['area']:.2e}`")
                
            with c_det2:
                st.markdown("**Van't Hoff Analysis (Integral)**")
                st.write(f"• Width ($\Delta T = A/H$): `{metrics['width_int']:.2f} K`")
                st.markdown(f"• $\Delta H_{{vH}}$: `{metrics['dh_vh']:.2e}`")


    # ==========================
    # TAB 2: Multi-Plot Comparison
    # ==========================
    with tab2:
        st.subheader("Multi-Plot Comparison")
        c1, c2 = st.columns([1, 4])
        
        with c1:
            show_raw = st.checkbox("Show Raw Cp", value=True)
            show_baseline = st.checkbox("Show Baseline", value=False)
            show_delta = st.checkbox("Show Delta Cp (Subtracted)", value=True)
            st.info("Uses Global Settings unless file is Locked.")

        with c2:
            comp_fig = go.Figure()
            
            for fname, raw_df in st.session_state.loaded_data.items():
                display_label = get_display_name(fname)
                
                if fname in st.session_state.locked_configs:
                    s = st.session_state.locked_configs[fname]
                else:
                    s = st.session_state.global_settings
                
                if s["peak_range"] is None: s = new_settings 

                grid_to_use = master_grid if (homogenize_data and 'master_grid' in locals()) else None
                proc_df = clean_and_standardize_data(raw_df, temp_col, cp_col, ref_grid=grid_to_use)
                
                if proc_df is not None and not proc_df.empty:
                    proc_df['Baseline'] = fit_baseline(
                        proc_df, s["method"], s["fit_regions"], 
                        s["smooth_factor"], s["poly_degree"], s["tm_manual"], s["sigmoid_width"]
                    )
                    proc_df['Delta_Cp'] = proc_df['Cp'] - proc_df['Baseline']
                    
                    if show_raw:
                        comp_fig.add_trace(go.Scatter(x=proc_df['Temperature'], y=proc_df['Cp'], mode='lines', name=f"{display_label} (Raw)", line=dict(width=style_config['curve_width']), opacity=0.7))
                    
                    if show_baseline:
                         comp_fig.add_trace(go.Scatter(x=proc_df['Temperature'], y=proc_df['Baseline'], mode='lines', name=f"{display_label} (Base)", line=dict(dash='dot', width=style_config['curve_width']), opacity=0.5))
                         
                    if show_delta:
                         comp_fig.add_trace(go.Scatter(x=proc_df['Temperature'], y=proc_df['Delta_Cp'], mode='lines', name=f"{display_label} (ΔCp)", line=dict(width=style_config['curve_width'])))
            
            comp_fig.update_layout(title="Comparative Analysis")
            
            # Apply Advanced Styling
            comp_fig = update_axes_style(comp_fig, style_config['x_label'], style_config['y_label'], style_config)
            
            st.plotly_chart(comp_fig, use_container_width=True)


    # ==========================
    # TAB 3: Batch Results
    # ==========================
    with tab3:
        st.subheader("Batch Analysis Report")
        
        if st.button("Recalculate Batch Results"):
            batch_results = []
            total_files = len(st.session_state.loaded_data)
            prog_bar = st.progress(0)
            
            for i, (fname, raw_df) in enumerate(st.session_state.loaded_data.items()):
                display_label = get_display_name(fname)
                try:
                    if fname in st.session_state.locked_configs:
                        s = st.session_state.locked_configs[fname]
                        used_settings = "Locked"
                    else:
                        s = st.session_state.global_settings
                        used_settings = "Global"
                    
                    if s["peak_range"] is None: s = new_settings

                    grid_to_use = master_grid if (homogenize_data and 'master_grid' in locals()) else None
                    proc_df = clean_and_standardize_data(raw_df, temp_col, cp_col, ref_grid=grid_to_use)
                    
                    if proc_df is not None and not proc_df.empty:
                        proc_df['Baseline'] = fit_baseline(
                            proc_df, s["method"], s["fit_regions"], 
                            s["smooth_factor"], s["poly_degree"], s["tm_manual"], s["sigmoid_width"]
                        )
                        proc_df['Delta_Cp'] = proc_df['Cp'] - proc_df['Baseline']
                        
                        m = calculate_metrics(proc_df, s["peak_range"][0], s["peak_range"][1], energy_unit)
                        
                        batch_results.append({
                            "Filename": fname,
                            "Display Name": display_label,
                            "Config Used": used_settings,
                            "Tm (°C)": round(m['tm'], 2),
                            "dH_cal (Area)": m['area'],
                            "dH_vH (Molar)": m['dh_vh'],
                            "Cooperativity (n)": m['coop_ratio'],
                            "Int. Start": s["peak_range"][0],
                            "Int. End": s["peak_range"][1],
                            "Status": "Success"
                        })
                    else:
                        batch_results.append({"Filename": fname, "Display Name": display_label, "Status": "Failed"})
                except Exception as e:
                    batch_results.append({"Filename": fname, "Display Name": display_label, "Status": f"Error: {str(e)}"})
                
                prog_bar.progress((i + 1) / total_files)
                
            res_df = pd.DataFrame(batch_results)
            st.dataframe(res_df, use_container_width=True)
            
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "dsc_batch_results.csv", "text/csv")
        else:
            st.info("Click 'Recalculate' to update table with latest settings.")

else:
    st.info("Upload data or use Mock Data to begin.")