# ProteinThermoToolkit

A small **Streamlit-based toolkit** for analysing **protein thermal unfolding thermodynamics** from a two-state model and for estimating **DSC cooperativity** from baseline-subtracted calorimetric peaks.

The repository currently contains **two independent apps**:

1. **Stability & thermodynamic visualisation** (`Gibbs_calculate.py`)  
   Reconstructs and visualises temperature-dependent **ΔG(T)**, **ΔH(T)**, **TΔS(T)**, **fU(T)**, and a simulated **DSC-like C\_p,excess(T)** profile.

2. **DSC peak integration & cooperativity** (`Hm_calculate.py`)  
   Provides interactive baseline selection, integrates **ΔC\_p(T)** to obtain **ΔH_cal**, estimates **ΔH_vH**, and reports **cooperativity** as a diagnostic ratio.

---

## What this toolkit is for

Thermal unfolding data are often reported as a small set of parameters (e.g., *T_m*, ΔH). However, many practical questions require **full temperature-dependent curves** and **uncertainty estimates**:

- How does ΔG(T) change across a wide temperature range?
- How sensitive are derived quantities (*T_s*, *T_c*, ΔG at 25 °C) to parameter uncertainties?
- Does a DSC transition look consistent with a **highly cooperative** (near two-state) unfolding event?

This toolkit focuses on these questions with an interactive workflow.

---

## Module 1 — Thermal Denaturation Tool (`Gibbs_calculate.py`)

### Model
Implements a modified Gibbs–Helmholtz framework with a **temperature-dependent heat capacity**:

\[
\Delta C_p(T) = \Delta C_{p,m} + \alpha (T - T_m)
\]

From this, the app computes (as functions of temperature):

- **ΔH(T)** and **TΔS(T)**
- **ΔG(T) = ΔH(T) − TΔS(T)**
- **Fraction unfolded, f_U(T)** from a two-state equilibrium
- A simulated **C\_p,excess(T)** (DSC-like) profile

### Key features
- Multi-variant comparison (WT vs mutants/conditions)
- Monte Carlo propagation of parameter uncertainties into curve envelopes
- Derived characteristic temperatures: **T_m**, **T_s** (maximum stability), **T_c** (cold denaturation, when present)
- Export of curves and summary statistics
- Save/load analysis state as JSON

---

## Module 2 — DSC Peak Integrator (`Hm_calculate.py`)

This app is designed for **DSC thermograms** (or any temperature vs C_p trace where a transition peak is present).

### Core quantities
1. **Calorimetric enthalpy**
\[
\Delta H_{cal} = \int_{T_1}^{T_2} \Delta C_p(T)\, dT
\]
where ΔC_p(T) is the **baseline-subtracted** peak.

2. **Van’t Hoff enthalpy (integral-width estimate)**
The app uses a peak-height/area relationship to estimate **ΔH_vH** (via \(C_{p,\max}\), \(T_m\), and the integrated area).

3. **Cooperativity diagnostic**
Reported as:
\[
n = \frac{\Delta H_{cal}}{\Delta H_{vH}}
\]
A value **near 1** is typically consistent with a strongly cooperative, near two-state transition under the chosen baselining and integration limits. Systematic deviations can indicate baseline artefacts, overlapping transitions, intermediates, or irreversible effects (interpretation depends on the system and data quality).

### Key features
- Upload **CSV/XLSX** thermograms
- Interactive peak-range selection
- Multiple baseline strategies (including a T_m-based sigmoidal baseline option)
- Multi-file comparison plots
- Per-file “lock” of settings for batch comparison

---

## Getting started (recommended: Miniconda on Windows)

### 1) Install Miniconda
Download and install **Miniconda3 (Windows 64-bit)**, then open:
**“Anaconda Prompt (Miniconda3)”**

### 2) Create and activate an environment
```bash
conda create -n thermo_env python=3.11
conda activate thermo_env
