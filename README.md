# ProteinThermoToolkit

A **Streamlit-based toolkit** for analysing **protein thermal unfolding thermodynamics** using a **two-state framework**, and for estimating **DSC cooperativity** from **baseline-subtracted calorimetric peaks**.

## What’s inside

This repository contains two independent Streamlit apps. The first app, Stability & thermodynamic visualisation (Gibbs_calculate.py), reconstructs and visualises temperature-dependent $\Delta G(T)$, $\Delta H(T)$, $T\Delta S(T)$, $f_U(T)$, and a simulated DSC-like $C_{p,\mathrm{excess}}(T)$ profile. It includes temperature-dependent $\Delta C_p$ and Monte Carlo uncertainty propagation to generate error envelopes. The second app, DSC peak integration & cooperativity (Hm_calculate.py), provides interactive baseline selection and integrates $\Delta C_p(T)$ to obtain $\Delta H_{cal}$, estimates $\Delta H_{vH}$, and reports cooperativity as a diagnostic ratio $n = \Delta H_{cal} / \Delta H_{vH}$.

## Why this toolkit exists

Thermal unfolding data are often summarised as a small set of parameters (e.g., $T_m$, $\Delta H$). In practice, many analyses require full temperature-dependent curves and uncertainty estimates. Typical questions include how $\Delta G(T)$ varies across a broad temperature range, how sensitive derived quantities such as $T_s$, $T_c$, and $\Delta G$ at 25 °C are to parameter uncertainty, and whether a DSC transition behaves like a highly cooperative (near two-state) unfolding event. This toolkit aims to make these analyses fast, transparent, and interactive.

## Module 1 — Thermal denaturation tool (`Gibbs_calculate.py`)

This module implements a modified Gibbs–Helmholtz framework with temperature-dependent heat capacity:

$$\Delta C_p(T) = \Delta C_{p,m} + \alpha (T - T_m)$$

where ($\Delta C_{p,m}$) is the heat capacity change at ($T_m$) and ($\alpha$) is the slope of the temperature dependence. From this, enthalpy and entropy are computed as functions of temperature:

$$\Delta H(T) = \Delta H_m + \int_{T_m}^{T} \Delta C_p(T)\, dT$$

$$\Delta S(T) = \frac{\Delta H_m}{T_m} + \int_{T_m}^{T} \frac{\Delta C_p(T)}{T}\, dT$$

and the Gibbs free energy is:

$$\Delta G(T) = \Delta H(T) - T\Delta S(T)$$

Key features include multi-variant comparison (e.g., WT vs mutants/conditions), Monte Carlo error propagation to visualise uncertainty bands, automatic computation of derived temperatures (($T_m$), ($T_s$) for maximum stability, and ($T_c$) for cold denaturation when present), export of curve data and summary tables (CSV), session save/load (JSON), and plot controls suitable for publication figures.

## Module 2 — DSC peak integrator (`Hm_calculate.py`)

This app is designed for DSC thermograms (or any ($T$) vs ($C_p$) trace with a transition peak). The calorimetric enthalpy is computed from baseline-subtracted data as:

$$\Delta H_{cal} = \int_{T_1}^{T_2} \Delta C_p(T)\, dT$$

where ($\Delta C_p(T)$) is the baseline-subtracted peak. The app also estimates van’t Hoff enthalpy (($\Delta H_{vH}$)) using a peak-height/area relationship (via ($C_{p,\max}$), ($T_m$), and the integrated peak), and reports the cooperativity diagnostic ($n$):

$$n = \frac{\Delta H_{cal}}{\Delta H_{vH}}$$

A value near 1 is typically consistent with a strongly cooperative, near two-state transition under the chosen baseline model and integration limits. Deviations can arise from baseline artefacts, overlapping transitions, intermediates, or irreversible processes. Key features include upload of CSV/XLSX thermograms, interactive peak-range selection, multiple baseline strategies (including a ($T_m$)-based sigmoidal baseline option), multi-file comparison plots, and per-file “lock” settings for batch analysis.

## Getting started (Windows + Miniconda)

Miniconda is recommended because it is lightweight and helps avoid dependency conflicts. Install Miniconda3 (Windows 64-bit), then open **Anaconda Prompt (Miniconda3)**. Create and activate a clean environment using:

```bash
conda create -n thermo_env python=3.11
conda activate thermo_env
```

Install dependencies using:

```bash
pip install streamlit pandas numpy plotly scipy openpyxl
```

Here, `scipy` is required for integration and peak calculations, and `openpyxl` enables reading `.xlsx` files via pandas. Navigate to the repository folder (for example, `E:\python_projects\ProteinThermoToolkit`) and run the apps:

```bash
streamlit run Gibbs_calculate.py
```

and

```bash
streamlit run Hm_calculate.py
```

Streamlit should open a browser tab automatically; if it does not, the terminal prints a local URL (typically `http://localhost:8501`).

## Input expectations

For `Gibbs_calculate.py`, parameters are entered in the UI per variant, with optional standard deviations for Monte Carlo uncertainty propagation. Temperature is handled internally in Kelvin (UI shown in °C). Energy units follow common protein-folding conventions (e.g., kcal/mol). For `Hm_calculate.py`, upload a CSV or XLSX file with temperature and heat capacity columns. The preferred column names are `Temperature` and `Cp`; if these are not found, the first two columns are used. Temperature is assumed in °C. Energy units should be chosen so that ($\Delta H_{cal}$), ($\Delta H_{vH}$), and the gas constant ($R$) are consistent.

## Suggested workflow

Extract thermal parameters from DSC fitting or literature (($T_m$), ($\Delta H_m$), ($\Delta C_{p,m}$), and optional ($\alpha$)). Use `Gibbs_calculate.py` to visualise ($\Delta G(T)$), compare variants, and inspect uncertainty envelopes. Use `Hm_calculate.py` on raw DSC thermograms to compute ($\Delta H_{cal}$), estimate ($\Delta H_{vH}$), and compare cooperativity across conditions.
