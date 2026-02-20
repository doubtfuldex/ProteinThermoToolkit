import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Physical constants ---
R_KJ = 0.008314       # kJ mol^-1 K^-1
KJ_TO_KCAL = 0.239006
KCAL_TO_KJ = 4.184
T_REF_K = 298.15      # 25 °C

# Default initial colors for reset/init
DEFAULT_COLORS = [
    "#2563eb",  # Blue
    "#dc2626",  # Red
    "#059669",  # Green
    "#d97706",  # Amber
    "#9333ea",  # Purple
    "#db2777",  # Pink
]

# --- Helper: Hex to RGBA for shaded areas ---
def hex_to_rgba(hex_color, opacity=0.2):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{opacity})"
    return f"rgba(0,0,0,{opacity})"

# --- Core thermodynamic calculations (Vectorized) ---

def calculate_thermo_arrays(temps_K, Tm_K, dHm_kJ, dCp_kJ_per_K, alpha_kJ_per_K2):
    """
    Calculates thermodynamic parameters for arrays of inputs (Monte Carlo) 
    or single scalars.
    
    Includes temperature-dependent Heat Capacity:
    dCp(T) = dCp_m + alpha * (T - Tm)
    
    Returns H (kJ), TS (kJ), dG (kJ), fu (0-1), Cp_excess (kJ/K).
    """
    dt = temps_K - Tm_K

    # 1. Enthalpy H(T)
    H_kJ = dHm_kJ + dCp_kJ_per_K * dt + 0.5 * alpha_kJ_per_K2 * (dt**2)

    # 2. Entropy S(T)
    dSm_kJ_per_K = dHm_kJ / Tm_K
    
    # Note: Using abs for log to prevent warnings during random sampling if weird values occur
    term_log = np.log(temps_K / Tm_K)
    
    S_kJ_per_K = dSm_kJ_per_K + (dCp_kJ_per_K - alpha_kJ_per_K2 * Tm_K) * term_log + alpha_kJ_per_K2 * dt
    
    TS_kJ = temps_K * S_kJ_per_K

    # 3. Free Energy ΔG(T) = H(T) − TΔS(T)
    dG_kJ = H_kJ - TS_kJ

    # 4. Fraction unfolded
    exponent = dG_kJ / (R_KJ * temps_K)
    
    # Sigmoid function with overflow protection
    fu = np.zeros_like(exponent)
    mask_mid = (exponent > -100) & (exponent < 100)
    
    if mask_mid.shape == exponent.shape:
        fu[mask_mid] = 1.0 / (1.0 + np.exp(exponent[mask_mid]))
        fu[exponent <= -100] = 1.0
        fu[exponent >= 100] = 0.0

    # 5. Excess Heat Capacity (DSC Profile)
    dCp_at_T = dCp_kJ_per_K + alpha_kJ_per_K2 * dt
    
    # Peak contribution (Van't Hoff)
    term_peak = (H_kJ**2) / (R_KJ * temps_K**2) * fu * (1.0 - fu)
    
    # Baseline contribution
    term_baseline = fu * dCp_at_T
    
    Cp_excess_kJ = term_peak + term_baseline
    
    return H_kJ, TS_kJ, dG_kJ, fu, Cp_excess_kJ

def compute_curves(variants, t_min_C, t_max_C, step_C, n_simulations=500):
    """
    Compute H, TΔS, ΔG, fraction unfolded, and Cp excess.
    Includes Monte Carlo error propagation if errors are present.
    """
    temps_C = np.arange(t_min_C, t_max_C + step_C, step_C, dtype=float)
    temps_K = temps_C + 273.15
    
    results = {}
    summary_rows = []

    for v in variants:
        if not v.get("show", True):
            continue

        name = v["name"]
        color = v["color"]
        
        # Nominal values
        Tm_C = v["Tm_C"]
        Tm_K = Tm_C + 273.15
        dHm_kJ = v["dHm_kcal"] * KCAL_TO_KJ
        dCp_kJ = v["dCp_kcal"] * KCAL_TO_KJ
        alpha_kJ = v.get("alpha_kcal", 0.0) * KCAL_TO_KJ
        
        # Error values (Standard Deviation)
        err_Tm = v.get("err_Tm", 0.0)
        err_dHm_kJ = v.get("err_dHm", 0.0) * KCAL_TO_KJ
        err_dCp_kJ = v.get("err_dCp", 0.0) * KCAL_TO_KJ
        err_alpha_kJ = v.get("err_alpha", 0.0) * KCAL_TO_KJ

        # Check if we need simulation
        has_error = (err_Tm > 0 or err_dHm_kJ > 0 or err_dCp_kJ > 0 or err_alpha_kJ > 0)
        
        v_err_dH_25C = 0.0
        v_err_dS_25C = 0.0
        v_err_dG_25C = 0.0
        v_err_Ts = 0.0
        v_err_Tc = 0.0
        
        # 1. Calculate Nominal Curve
        H_kJ, TS_kJ, dG_kJ, fu, Cp_kJ = calculate_thermo_arrays(temps_K, Tm_K, dHm_kJ, dCp_kJ, alpha_kJ)
        
        # Calculate Delta Cp (Theoretical Baseline: dCp + alpha*dt)
        dt_nominal = temps_K - Tm_K
        DeltaCp_kJ = dCp_kJ + alpha_kJ * dt_nominal

        # Store nominal results (converted to kcal)
        res_data = {
            "color": color,
            "H_kcal": H_kJ * KJ_TO_KCAL,
            "TS_kcal": TS_kJ * KJ_TO_KCAL,
            "dG_kcal": dG_kJ * KJ_TO_KCAL,
            "Cp_kcal": Cp_kJ * KJ_TO_KCAL,
            "DeltaCp_kcal": DeltaCp_kJ * KJ_TO_KCAL, # New column data
            "fu": fu,
            "has_error": has_error
        }
        
        # 2. Monte Carlo Simulation for Error Bands & Extrapolation Errors
        if has_error:
            # Generate random distributions (N_sim, 1)
            rng = np.random.default_rng(42)
            sim_Tm_K = rng.normal(Tm_K, err_Tm, (n_simulations, 1))
            sim_dHm_kJ = rng.normal(dHm_kJ, err_dHm_kJ, (n_simulations, 1))
            sim_dCp_kJ = rng.normal(dCp_kJ, err_dCp_kJ, (n_simulations, 1))
            sim_alpha_kJ = rng.normal(alpha_kJ, err_alpha_kJ, (n_simulations, 1))
            
            # --- Curve Simulation ---
            sim_temps_K = temps_K.reshape(1, -1)
            
            # Calculate arrays (N_sim, n_temps)
            s_H, s_TS, s_dG, s_fu, s_Cp = calculate_thermo_arrays(
                sim_temps_K, sim_Tm_K, sim_dHm_kJ, sim_dCp_kJ, sim_alpha_kJ
            )
            
            # Calculate Delta Cp simulation for uncertainty
            dt_sim = sim_temps_K - sim_Tm_K
            s_DeltaCp = sim_dCp_kJ + sim_alpha_kJ * dt_sim

            # Calculate standard deviations (1 sigma)
            std_H = np.std(s_H, axis=0) * KJ_TO_KCAL
            std_TS = np.std(s_TS, axis=0) * KJ_TO_KCAL
            std_dG = np.std(s_dG, axis=0) * KJ_TO_KCAL
            std_Cp = np.std(s_Cp, axis=0) * KJ_TO_KCAL
            std_fu = np.std(s_fu, axis=0)
            std_DeltaCp = np.std(s_DeltaCp, axis=0) * KJ_TO_KCAL
            
            res_data["H_std"] = std_H
            res_data["TS_std"] = std_TS
            res_data["dG_std"] = std_dG
            res_data["Cp_std"] = std_Cp
            res_data["fu_std"] = std_fu
            res_data["DeltaCp_std"] = std_DeltaCp
            
            # --- Extrapolation Error at 25C (298.15 K) ---
            dt_ref = T_REF_K - sim_Tm_K
            
            # Enthalpy
            sim_dH_ref_kJ = sim_dHm_kJ + sim_dCp_kJ * dt_ref + 0.5 * sim_alpha_kJ * (dt_ref**2)
            
            # Entropy
            term_log_ref = np.log(T_REF_K / sim_Tm_K)
            sim_dS_ref_kJ = (sim_dHm_kJ / sim_Tm_K) + (sim_dCp_kJ - sim_alpha_kJ * sim_Tm_K) * term_log_ref + sim_alpha_kJ * dt_ref
            
            # Delta G
            sim_dG_ref_kJ = sim_dH_ref_kJ - T_REF_K * sim_dS_ref_kJ
            
            # Std Devs
            err_dH_ref_kJ = np.std(sim_dH_ref_kJ)
            err_dS_ref_kJ = np.std(sim_dS_ref_kJ)
            err_dG_ref_kJ = np.std(sim_dG_ref_kJ)
            
            # Convert to display units
            v_err_dH_25C = err_dH_ref_kJ * KJ_TO_KCAL
            v_err_dS_25C = err_dS_ref_kJ * KJ_TO_KCAL * 1000 # to cal/mol/K
            v_err_dG_25C = err_dG_ref_kJ * KJ_TO_KCAL

            # --- Tc and Ts Statistics ---
            sim_Ts_vals = []
            sim_Tc_vals = []
            
            for i in range(n_simulations):
                y_dG_row = s_dG[i]
                this_Tm = sim_Tm_K[i][0] - 273.15
                
                # 1. Ts (Max Stability)
                idx_max = np.argmax(y_dG_row)
                if 0 < idx_max < len(temps_C) - 1:
                    x_sub = temps_C[idx_max-1 : idx_max+2]
                    y_sub = y_dG_row[idx_max-1 : idx_max+2]
                    p = np.polyfit(x_sub, y_sub, 2)
                    if p[0] != 0:
                        ts = -p[1] / (2 * p[0])
                        sim_Ts_vals.append(ts)
                    else:
                        sim_Ts_vals.append(temps_C[idx_max])
                else:
                    sim_Ts_vals.append(temps_C[idx_max])
                
                # 2. Tc (Cold Denaturation)
                mask_low = temps_C < this_Tm
                if np.any(mask_low):
                    y_low = y_dG_row[mask_low]
                    x_low = temps_C[mask_low]
                    signs = np.sign(y_low)
                    sign_change = (signs[:-1] * signs[1:]) < 0
                    indices = np.where(sign_change)[0]
                    
                    if len(indices) > 0:
                        idx_c = indices[-1]
                        x1, x2 = x_low[idx_c], x_low[idx_c+1]
                        y1, y2 = y_low[idx_c], y_low[idx_c+1]
                        if y2 != y1:
                            tc = x1 - y1 * (x2 - x1) / (y2 - y1)
                            sim_Tc_vals.append(tc)
                        else:
                            sim_Tc_vals.append(x1)
                    else:
                        sim_Tc_vals.append(np.nan)
                else:
                    sim_Tc_vals.append(np.nan)
            
            v_err_Ts = np.nanstd(sim_Ts_vals)
            v_err_Tc = np.nanstd(sim_Tc_vals)

        results[name] = res_data

        # --- Summary Stats Calculation (Nominal) ---
        cold_T_C = None
        below_Tm_idx = np.where(temps_K < Tm_K)[0]
        if len(below_Tm_idx) > 0:
            dG_below = dG_kJ[below_Tm_idx]
            temps_below = temps_C[below_Tm_idx]
            sign_changes = np.where(np.diff(np.sign(dG_below)))[0]
            if len(sign_changes) > 0:
                idx = sign_changes[0]
                y1, y2 = dG_below[idx], dG_below[idx+1]
                x1, x2 = temps_below[idx], temps_below[idx+1]
                if y2 != y1:
                    cold_T_C = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                else:
                    cold_T_C = x1

        max_dG_idx = np.argmax(dG_kJ)
        Ts_C_approx = temps_C[max_dG_idx]
        
        if 0 < max_dG_idx < len(temps_C) - 1:
            x_peak = temps_C[max_dG_idx-1:max_dG_idx+2]
            y_peak = dG_kJ[max_dG_idx-1:max_dG_idx+2]
            poly = np.polyfit(x_peak, y_peak, 2)
            Ts_C = -poly[1] / (2 * poly[0])
        else:
            Ts_C = Ts_C_approx
            
        tm_tc_diff = (v["Tm_C"] - cold_T_C) if cold_T_C is not None else None
        tm_tc_mean = ((v["Tm_C"] + cold_T_C) / 2.0) if cold_T_C is not None else None
        
        dt_ref = T_REF_K - Tm_K
        dH_ref_kJ = dHm_kJ + dCp_kJ * dt_ref + 0.5 * alpha_kJ * (dt_ref**2)
        term_log_ref = np.log(T_REF_K / Tm_K)
        dS_ref_kJ = (dHm_kJ / Tm_K) + (dCp_kJ - alpha_kJ * Tm_K) * term_log_ref + alpha_kJ * dt_ref
        dG_ref_kJ = dH_ref_kJ - T_REF_K * dS_ref_kJ
        dS_ref_cal = dS_ref_kJ * KJ_TO_KCAL * 1000 

        summary_rows.append({
            "name": name,
            "Tm_C": v["Tm_C"],
            "dHm_kcal": v["dHm_kcal"],
            "dH_25C_kcal": dH_ref_kJ * KJ_TO_KCAL,
            "dS_25C_cal": dS_ref_cal,
            "dG_25C_kcal": dG_ref_kJ * KJ_TO_KCAL,
            "dCp_kcal_per_C": v["dCp_kcal"],
            "alpha_kcal": v.get("alpha_kcal", 0.0),
            "cold_Tm_C": cold_T_C,
            "Ts_C": Ts_C,
            "Tm_minus_Tc": tm_tc_diff,
            "Mean_Tm_Tc": tm_tc_mean,
            "err_dH_25C": v_err_dH_25C,
            "err_dS_25C": v_err_dS_25C,
            "err_dG_25C": v_err_dG_25C,
            "err_Ts": v_err_Ts,
            "err_Tc": v_err_Tc
        })

    return temps_C, results, summary_rows


# --- Plotting helpers ---

def update_axes_style(fig, x_label, y_label, x_range, y_range, config):
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
    legend_bg_color = config.get("legend_bg_color", "rgba(255,255,255,0.8)") # Default to semi-transparent white
    
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
        showlegend=config["show_legend"],
        legend=legend_dict,
        font=dict(family=font_family, color="black"), 
        hovermode="x unified",
        margin=dict(l=80, r=40, t=50, b=80),
        # New Background Logic
        paper_bgcolor=config["bg_color"],
        plot_bgcolor=config["bg_color"],
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
    
    if x_range and len(x_range) == 2 and x_range[0] < x_range[1]:
        fig.update_xaxes(range=x_range)
    if y_range and len(y_range) == 2 and y_range[0] < y_range[1]:
        fig.update_yaxes(range=y_range)
    
    fig.update_layout(width=config["width"], height=config["height"])
    return fig

def add_shaded_trace(fig, x, y_nominal, y_std, color, name, config, dash=None, line_width=2):
    """Adds a line with a shaded error region (sweeping area)."""
    group_id = name 
    
    # Extract config
    opacity = config.get("error_opacity", 0.2)
    sigma_mult = config.get("sigma_mult", 1.0)

    if y_std is not None:
        y_upper = y_nominal + (y_std * sigma_mult)
        y_lower = y_nominal - (y_std * sigma_mult)
        fill_color = hex_to_rgba(color, opacity=opacity)
        
        # Determine name based on sigma
        sigma_label = f"{sigma_mult}σ"
        if sigma_mult == 1: sigma_label = "1σ (68%)"
        elif sigma_mult == 2: sigma_label = "2σ (95%)"
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=group_id, 
            name=f"{name} ±{sigma_label}"
        ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y_nominal,
        mode="lines",
        name=name,
        legendgroup=group_id, 
        line=dict(color=color, width=line_width, dash=dash),
        showlegend=config["show_legend"]
    ))


def make_enthalpy_figure(results, temps_C, config):
    fig = go.Figure()
    w = config.get("curve_width", 2)
    
    for name, data in results.items():
        if config["show_enthalpy"]:
            std = data.get("H_std") if config["show_errors"] else None
            add_shaded_trace(fig, temps_C, data["H_kcal"], std, data["color"], f"{name} ΔH", config, line_width=w)
            
        if config["show_TS"]:
            std = data.get("TS_std") if config["show_errors"] else None
            add_shaded_trace(fig, temps_C, data["TS_kcal"], std, data["color"], f"{name} TΔS", config, dash="dash", line_width=w)

    return update_axes_style(
        fig,
        "Temperature (°C)", "Energy (kcal/mol)",
        config["x_range"], config["y_range_H"],
        config
    )


def make_dG_figure(results, temps_C, config):
    fig = go.Figure()
    w = config.get("curve_width", 2)
    
    for name, data in results.items():
        std = data.get("dG_std") if config["show_errors"] else None
        add_shaded_trace(fig, temps_C, data["dG_kcal"], std, data["color"], name, config, line_width=w)
        
    fig.add_hline(y=0.0, line=dict(color="black", width=1, dash="dot"))

    return update_axes_style(
        fig,
        "Temperature (°C)", "ΔG (kcal/mol)",
        config["x_range"], config["y_range_G"],
        config
    )


def make_fu_figure(results, temps_C, config):
    fig = go.Figure()
    w = config.get("curve_width", 2)
    
    for name, data in results.items():
        std = data.get("fu_std") if config["show_errors"] else None
        add_shaded_trace(fig, temps_C, data["fu"], std, data["color"], name, config, line_width=w)

    y_range = config["y_range_fu"] if config["y_range_fu"] else [0.0, 1.05]

    return update_axes_style(
        fig,
        "Temperature (°C)", "Fraction Unfolded",
        config["x_range"], y_range,
        config
    )

def make_Cp_figure(results, temps_C, config):
    fig = go.Figure()
    w = config.get("curve_width", 2)
    
    for name, data in results.items():
        std = data.get("Cp_std") if config["show_errors"] else None
        add_shaded_trace(fig, temps_C, data["Cp_kcal"], std, data["color"], name, config, line_width=w)

    return update_axes_style(
        fig,
        "Temperature (°C)", "Excess Cp (kcal/mol/K)",
        config["x_range"], config["y_range_Cp"],
        config
    )

def make_ddG_figure(results, temps_C, ref_name, config):
    """
    Plots Delta Delta G (Phase Diagram).
    y = dG_var - dG_ref
    Positive = Stabilizing, Negative = Destabilizing.
    """
    fig = go.Figure()
    w = config.get("curve_width", 2)
    
    if ref_name not in results:
        fig.add_annotation(
            text=f"Reference variant '{ref_name}' is not visible.<br>Please enable it in the Sidebar.",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return update_axes_style(fig, "Temperature (°C)", "ΔΔG (kcal/mol)", config["x_range"], None, config)

    ref_data = results[ref_name]
    ref_dG = ref_data["dG_kcal"]
    ref_std = ref_data.get("dG_std", np.zeros_like(ref_dG)) if config["show_errors"] else None

    # Region Highlights
    if config.get("highlight_ddg", False):
        # Stabilizing Region (Above 0)
        fig.add_shape(type="rect",
            xref="paper", yref="y",
            x0=0, y0=0, x1=1, y1=10000, # Large number
            fillcolor="rgba(34, 197, 94, 0.1)", # Green
            line_width=0, layer="below"
        )
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper", 
            text="Stabilized (More Positive ΔG)", 
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(color="green", size=12)
        )

        # Destabilizing Region (Below 0)
        fig.add_shape(type="rect",
            xref="paper", yref="y",
            x0=0, y0=0, x1=1, y1=-10000,
            fillcolor="rgba(239, 68, 68, 0.1)", # Red
            line_width=0, layer="below"
        )
        fig.add_annotation(
            x=0.02, y=0.02, xref="paper", yref="paper", 
            text="Destabilized (Less Positive ΔG)", 
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(color="red", size=12)
        )

    for name, data in results.items():
        # Calculate Delta Delta G
        ddG = data["dG_kcal"] - ref_dG
        
        # Propagate error: sqrt(sigma_var^2 + sigma_ref^2)
        # For the reference itself, difference is exactly 0.
        std = None
        if config["show_errors"] and data.get("has_error"):
            var_std = data["dG_std"]
            if name == ref_name:
                std = np.zeros_like(var_std)
            else:
                std = np.sqrt(var_std**2 + ref_std**2)

        add_shaded_trace(fig, temps_C, ddG, std, data["color"], name, config, line_width=w)

    fig.add_hline(y=0.0, line=dict(color="black", width=2))

    return update_axes_style(
        fig,
        "Temperature (°C)", "ΔΔG (kcal/mol)",
        config["x_range"], config.get("y_range_ddG"),
        config
    )

# --- Session Management ---

def get_current_state():
    state = {}
    keys = [
        "sim_min", "sim_max", "sim_step", "fig_w", "fig_h", "n_var",
        "cx_check", "cx_min", "cx_max",
        "cyh_check", "cyh_min", "cyh_max",
        "cyg_check", "cyg_min", "cyg_max",
        "cyf_check", "cyf_min", "cyf_max",
        "cyc_check", "cyc_min", "cyc_max",
        "show_leg", "show_H", "show_TS", "show_err_env", 
        "leg_pos_sel", "x_dtick", "y_dtick", "minor_divs",
        "n_sims", 
        "axis_thick", "axis_col", "show_grid", "grid_col",
        "font_fam", "tick_font_sz", "label_col", "title_font_sz", "title_col",
        "maj_tick_len", "maj_tick_wid", "maj_tick_col",
        "show_minor", "min_tick_len", "min_tick_wid", "min_tick_col",
        "curve_wid", "show_leg_border",
        "ref_var_idx", "highlight_ddg", "cyddg_check", "cyddg_min", "cyddg_max",
        "err_opacity", "sigma_mult",
        "trans_bg", "bg_col",
        "trans_leg_bg", "leg_bg_col", "leg_bg_opacity" # New keys for legend background
    ]
    for k in keys:
        if k in st.session_state:
            state[k] = st.session_state[k]
            
    if "n_var" in st.session_state:
        for i in range(st.session_state["n_var"]):
            v_keys = [
                f"v_name_{i}", f"v_show_{i}", f"v_col_{i}",
                f"v_tm_{i}", f"v_dh_{i}", f"v_dcp_{i}",
                f"v_alpha_{i}",
                f"v_err_tm_{i}", f"v_err_dh_{i}", f"v_err_dcp_{i}", f"v_err_alpha_{i}"
            ]
            for vk in v_keys:
                if vk in st.session_state:
                    state[vk] = st.session_state[vk]
    return state

def load_state_from_json(json_data):
    try:
        data = json.load(json_data)
        for k, v in data.items():
            st.session_state[k] = v
        return True
    except Exception as e:
        st.error(f"Error loading session: {e}")
        return False

# --- Main app ---

def main():
    st.set_page_config(page_title="Thermal Denaturation Tool", layout="wide")

    st.title("Protein Thermal Stability Analysis")
    st.markdown(
        """
        Model two-state protein folding thermodynamics with temperature-dependent Heat Capacity.
        $$\Delta C_p(T) = \Delta C_{p,m} + \\alpha(T - T_m)$$
        """
    )

    with st.sidebar:
        st.header("Configuration")

        # --- Session Manager ---
        with st.expander("💾 Session Manager", expanded=False):
            uploaded_file = st.file_uploader("Load Session", type=["json"])
            if uploaded_file and st.button("Apply Loaded Session"):
                if load_state_from_json(uploaded_file):
                    st.success("Loaded! UI updating...")
                    st.rerun()

            st.markdown("---")
            if st.button("Save Current Session"):
                current_state = get_current_state()
                json_str = json.dumps(current_state, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="thermo_session.json",
                    mime="application/json"
                )

        st.markdown("---")
        
        tab_global, tab_variants, tab_axes = st.tabs(["Global", "Variants", "Axes/Style"])

        # Global settings
        with tab_global:
            st.subheader("Simulation")
            t_min_C = st.number_input("Min Temp (°C)", value=-20.0, step=10.0, key="sim_min")
            t_max_C = st.number_input("Max Temp (°C)", value=120.0, step=10.0, key="sim_max")
            step_C = st.number_input("Step Size (°C)", value=1.0, min_value=0.1, key="sim_step")
            
            st.markdown("**Comparison Settings**")
            # We use an index based selector to allow dynamic naming later
            ref_var_idx = st.number_input(
                "Reference Variant #", 
                min_value=1, 
                max_value=20, # arbitrary max
                value=1, 
                key="ref_var_idx",
                help="Select which variant number (1, 2, 3...) serves as the baseline for ΔΔG calculations."
            )

            st.markdown("**Monte Carlo Settings**")
            n_sims = st.number_input(
                "Monte Carlo Iterations", 
                value=500, min_value=10, max_value=100000, step=100, 
                key="n_sims"
            )
            
            st.subheader("Options")
            show_err_env = st.checkbox("Show Error Envelopes", value=True, key="show_err_env")
            # Added Distribution Control
            sigma_mult = st.slider(
                "Confidence Interval (σ)", 
                min_value=0.5, max_value=3.0, value=1.0, step=0.1, 
                key="sigma_mult",
                help="Scaling factor for standard deviation. 1σ ≈ 68%, 2σ ≈ 95%, 3σ ≈ 99%."
            )

        # Variants
        with tab_variants:
            num_variants = st.number_input("Count", 1, 8, 2, key="n_var")
            variants = []
            
            for i in range(num_variants):
                with st.expander(f"Variant {i+1}", expanded=(i == 0)):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        v_name = st.text_input("Name", value=f"Variant {i+1}", key=f"v_name_{i}")
                    with c2:
                        default_col = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                        v_color = st.color_picker("Color", value=default_col, key=f"v_col_{i}")
                    with c3:
                        v_show = st.checkbox("Show", value=True, key=f"v_show_{i}")
                    
                    st.markdown("**Parameters**")
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1:
                        v_Tm = st.number_input("Tm (°C)", value=60.0 + (i * 5), step=1.0, key=f"v_tm_{i}")
                    with col_p2:
                        v_dHm = st.number_input("ΔH(Tm) (kcal)", value=100.0, step=5.0, key=f"v_dh_{i}")
                    with col_p3:
                        v_dCp = st.number_input("ΔCp (kcal/K)", value=2.0, step=0.1, format="%.2f", key=f"v_dcp_{i}")

                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        v_alpha = st.number_input("α (kcal/K²)", value=0.0, step=0.01, format="%.4f", key=f"v_alpha_{i}")

                    st.markdown("**Uncertainty (± 1σ)**")
                    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                    with col_e1:
                        v_err_Tm = st.number_input("± Tm", value=0.0, step=0.1, key=f"v_err_tm_{i}")
                    with col_e2:
                        v_err_dHm = st.number_input("± ΔH", value=0.0, step=1.0, key=f"v_err_dh_{i}")
                    with col_e3:
                        v_err_dCp = st.number_input("± ΔCp", value=0.0, step=0.05, key=f"v_err_dcp_{i}")
                    with col_e4:
                        v_err_alpha = st.number_input("± α", value=0.0, step=0.01, format="%.4f", key=f"v_err_alpha_{i}")

                    variants.append({
                        "name": v_name, "color": v_color, "show": v_show,
                        "Tm_C": v_Tm, "dHm_kcal": v_dHm, "dCp_kcal": v_dCp,
                        "alpha_kcal": v_alpha,
                        "err_Tm": v_err_Tm, "err_dHm": v_err_dHm, "err_dCp": v_err_dCp, "err_alpha": v_err_alpha
                    })

        # Axes / Style
        with tab_axes:
            
            with st.expander("Layout & Dimensions", expanded=True):
                c_fig1, c_fig2 = st.columns(2)
                canvas_width = c_fig1.number_input("Width (px)", 400, 4000, 800, 50, key="fig_w")
                canvas_height = c_fig2.number_input("Height (px)", 300, 4000, 600, 50, key="fig_h")
            
            with st.expander("Background Styling", expanded=True):
                transparent_bg = st.checkbox("Transparent Background", value=False, key="trans_bg")
                if not transparent_bg:
                    bg_col_val = st.color_picker("Background Color", "#FFFFFF", key="bg_col")
                else:
                    bg_col_val = "rgba(0,0,0,0)"

            with st.expander("Curve Styling", expanded=True):
                curve_width = st.slider("Curve Line Width", 1, 10, 2, key="curve_wid")
                # Added Opacity Control
                err_opacity = st.slider("Error Shading Opacity", 0.0, 1.0, 0.2, 0.05, key="err_opacity")
                highlight_ddg = st.checkbox("Highlight Stability Regions (ΔΔG)", value=True, key="highlight_ddg")
            
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
                
                # New controls for Legend Background
                transparent_leg_bg = st.checkbox("Transparent Legend Background", False, key="trans_leg_bg")
                if not transparent_leg_bg:
                    leg_bg_col_val = st.color_picker("Legend Background Color", "#FFFFFF", key="leg_bg_col")
                    leg_bg_opacity = st.slider("Legend Background Opacity", 0.0, 1.0, 0.8, 0.1, key="leg_bg_opacity")
                    legend_bg_color = hex_to_rgba(leg_bg_col_val, leg_bg_opacity)
                else:
                    legend_bg_color = "rgba(0,0,0,0)"

                legend_options = [
                    "Outside Right", "Inside Top-Right", "Inside Top-Left", 
                    "Inside Bottom-Right", "Inside Bottom-Left"
                ]
                legend_pos = st.selectbox("Position", legend_options, index=0, key="leg_pos_sel")

            
            st.markdown("---")
            st.caption("Custom Ranges (Leave unchecked for auto)")
            
            # X-axis
            custom_x = st.checkbox("Custom X-Axis", key="cx_check")
            x_range = None
            if custom_x:
                c1, c2 = st.columns(2)
                xm = c1.number_input("Min X", value=float(t_min_C), key="cx_min")
                xM = c2.number_input("Max X", value=float(t_max_C), key="cx_max")
                if xm < xM: x_range = [xm, xM]

            # Y-axis Enthalpy
            custom_yH = st.checkbox("Custom Y (Enthalpy)", key="cyh_check")
            y_range_H = None
            if custom_yH:
                c1, c2 = st.columns(2)
                yHm = c1.number_input("Min H", value=-50.0, key="cyh_min")
                yHM = c2.number_input("Max H", value=200.0, key="cyh_max")
                if yHm < yHM: y_range_H = [yHm, yHM]

            # Y-axis dG
            custom_yG = st.checkbox("Custom Y (ΔG)", key="cyg_check")
            y_range_G = None
            if custom_yG:
                c1, c2 = st.columns(2)
                yGm = c1.number_input("Min G", value=-20.0, key="cyg_min")
                yGM = c2.number_input("Max G", value=20.0, key="cyg_max")
                if yGm < yGM: y_range_G = [yGm, yGM]

            # Y-axis Fraction
            custom_yF = st.checkbox("Custom Y (Fraction)", key="cyf_check")
            y_range_fu = None
            if custom_yF:
                c1, c2 = st.columns(2)
                yFm = c1.number_input("Min F", value=-0.1, key="cyf_min")
                yFM = c2.number_input("Max F", value=1.1, key="cyf_max")
                if yFm < yFM: y_range_fu = [yFm, yFM]

            # Y-axis Cp
            custom_yCp = st.checkbox("Custom Y (Cp)", key="cyc_check")
            y_range_Cp = None
            if custom_yCp:
                c1, c2 = st.columns(2)
                yCpm = c1.number_input("Min Cp", value=0.0, key="cyc_min")
                yCpM = c2.number_input("Max Cp", value=20.0, key="cyc_max")
                if yCpm < yCpM: y_range_Cp = [yCpm, yCpM]

            # Y-axis DDG
            custom_yDDG = st.checkbox("Custom Y (ΔΔG)", key="cyddg_check")
            y_range_ddG = None
            if custom_yDDG:
                c1, c2 = st.columns(2)
                yDDGm = c1.number_input("Min ΔΔG", value=-5.0, key="cyddg_min")
                yDDGM = c2.number_input("Max ΔΔG", value=5.0, key="cyddg_max")
                if yDDGm < yDDGM: y_range_ddG = [yDDGm, yDDGM]
            
            st.markdown("---")
            show_H_curve = st.checkbox("Show ΔH Trace", True, key="show_H")
            show_TS_curve = st.checkbox("Show TΔS Trace", True, key="show_TS")

    # Determine reference variant from index
    # We constrain the index to valid range
    safe_ref_idx = max(0, min(ref_var_idx - 1, len(variants) - 1))
    # Note: If the user inputs a number higher than count, it defaults to last one.
    if len(variants) > 0:
        ref_variant_name = variants[safe_ref_idx]["name"]
    else:
        ref_variant_name = "None"

    # Compute curves with timer
    start_time = time.time()
    temps_C, results, summary_rows = compute_curves(variants, t_min_C, t_max_C, step_C, n_simulations=n_sims)
    end_time = time.time()
    elapsed = end_time - start_time

    if n_sims > 0 and any(v["show"] for v in variants):
        st.caption(f"Calculated {n_sims} simulations in {elapsed:.3f} seconds. Ref: {ref_variant_name}")

    if not results:
        st.warning("No variants enabled. Enable a variant in the sidebar to visualize data.")
        return

    plot_config = {
        "width": canvas_width, "height": canvas_height,
        "x_range": x_range, "y_range_H": y_range_H, "y_range_G": y_range_G, "y_range_fu": y_range_fu, "y_range_Cp": y_range_Cp,
        "y_range_ddG": y_range_ddG,
        "show_legend": show_legend, "show_enthalpy": show_H_curve, "show_TS": show_TS_curve,
        "show_errors": show_err_env,
        "legend_pos": legend_pos,
        "highlight_ddg": highlight_ddg, 
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
        "error_opacity": err_opacity, # New
        "sigma_mult": sigma_mult, # New
        "transparent_bg": transparent_bg,
        "bg_color": bg_col_val,
        "legend_bg_color": legend_bg_color # New
    }

    col_main, col_data = st.columns([2.5, 1])

    with col_main:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stability (ΔG)", "Enthalpy Analysis", "Fraction Unfolded", "DSC Profile", "ΔΔG Analysis"])

        with tab1:
            fig_G = make_dG_figure(results, temps_C, plot_config)
            st.plotly_chart(fig_G, use_container_width=True)
            
        with tab2:
            fig_H = make_enthalpy_figure(results, temps_C, plot_config)
            st.plotly_chart(fig_H, use_container_width=True)

        with tab3:
            fig_fu = make_fu_figure(results, temps_C, plot_config)
            st.plotly_chart(fig_fu, use_container_width=True)
        
        with tab4:
            fig_Cp = make_Cp_figure(results, temps_C, plot_config)
            st.plotly_chart(fig_Cp, use_container_width=True)

        with tab5:
            st.info(f"Phase Diagram relative to Reference: **{ref_variant_name}**")
            fig_ddG = make_ddG_figure(results, temps_C, ref_variant_name, plot_config)
            st.plotly_chart(fig_ddG, use_container_width=True)

    with col_data:
        st.subheader("Statistics")
        df = pd.DataFrame(summary_rows)

        # Main Table
        display_df = df[["name", "Tm_C", "dHm_kcal", "dCp_kcal_per_C", "alpha_kcal"]].copy()
        display_df.columns = ["Variant", "Tm", "ΔH(Tm)", "ΔCp", "α"]
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        # Details
        st.markdown("##### Detailed Analysis")
        detail_df = df[["name", "Tm_minus_Tc", "cold_Tm_C"]].copy()
        detail_df.columns = ["Variant", "Tm - Tc", "Tc (Cold)"]
        
        cols_signed = ["Tm - Tc"]
        for col in cols_signed:
            detail_df[col] = detail_df[col].apply(lambda x: f"{x:+.1f}" if pd.notnull(x) else "-")
            
        def fmt_err(val, err):
            if pd.isna(val): return "-"
            if err == 0: return f"{val:.1f}"
            return f"{val:.1f} ± {err:.1f}"

        detail_df["ΔG(25°C)"] = df.apply(lambda x: fmt_err(x["dG_25C_kcal"], x["err_dG_25C"]), axis=1)
        detail_df["ΔH(25°C)"] = df.apply(lambda x: fmt_err(x["dH_25C_kcal"], x["err_dH_25C"]), axis=1)
        detail_df["ΔS(25°C)"] = df.apply(lambda x: fmt_err(x["dS_25C_cal"], x["err_dS_25C"]), axis=1)
        
        detail_df["Tc (Cold)"] = df.apply(lambda x: fmt_err(x["cold_Tm_C"], x["err_Tc"]), axis=1)
        detail_df["Ts (Max)"] = df.apply(lambda x: fmt_err(x["Ts_C"], x["err_Ts"]), axis=1)
        
        st.caption("Note: ΔS(25°C) is in cal/mol/K. ΔG and ΔH are in kcal/mol.")
            
        st.dataframe(detail_df, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("**Export Data**")
        
        # 1. Summary CSV
        csv_summary = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Statistics (CSV)", csv_summary, "thermo_summary.csv", "text/csv")

        # 2. Curves CSV
        curve_data = {"Temperature (°C)": temps_C}
        for name, data in results.items():
            curve_data[f"{name} - ΔG"] = data["dG_kcal"]
            curve_data[f"{name} - ΔH"] = data["H_kcal"]
            curve_data[f"{name} - TΔS"] = data["TS_kcal"]
            curve_data[f"{name} - Fraction"] = data["fu"]
            curve_data[f"{name} - Excess Cp"] = data["Cp_kcal"]
            curve_data[f"{name} - ΔCp"] = data["DeltaCp_kcal"]
            if data.get("has_error"):
                curve_data[f"{name} - ΔG (std)"] = data["dG_std"]
                curve_data[f"{name} - ΔH (std)"] = data["H_std"]
                curve_data[f"{name} - TΔS (std)"] = data["TS_std"]
                curve_data[f"{name} - Fraction (std)"] = data["fu_std"]
                curve_data[f"{name} - Excess Cp (std)"] = data["Cp_std"]
                curve_data[f"{name} - ΔCp (std)"] = data["DeltaCp_std"]
                
        df_curves = pd.DataFrame(curve_data)
        csv_curves = df_curves.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download Curves (CSV)", csv_curves, "thermo_curves.csv", "text/csv")

if __name__ == "__main__":
    main()