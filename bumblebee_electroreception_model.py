"""
Numerical analysis for:
"Ideal-observer limits on electric field detection by bumblebee hairs 
under environmental noise and mains interference"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import os

# Set up publication-quality plots (JEB requires Arial/Helvetica)
rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

print(f"\n---> LAS FIGURAS SE GUARDARÁN EN ESTA CARPETA: {os.getcwd()} <---\n")

# ============================================================
# 1. PHYSICAL AND MORPHOLOGICAL PARAMETERS
# ============================================================
I = 1.13e-17          # Moment of inertia (kg·m²)
f0 = 150.0            # Resonance frequency (Hz)
omega0 = 2 * np.pi * f0
k = I * omega0**2     # Torsional stiffness (N·m/rad)
Q = 2.0               # Quality factor
gamma = np.sqrt(k * I) / Q # Damping coefficient (N·m·s/rad)

p = 1.0e-15           # Effective dipole moment (C·m)
kappa_L = 8.6e-12     # Aerodynamic coupling (N·m/(m/s))

NE0 = 1.0e4           # Electrical noise floor: (V/m)²/Hz
f_c = 10.0            # Corner frequency for 1/f noise (Hz)
alpha = 1.0           # Spectral index for electrical noise
N_u = 1.35e-4         # Mechanical (air flow) noise: (m/s)²/Hz

f_s = 10.0            # Biological signal frequency (Hz)
f_m = 50.0            # Mains interference frequency (Hz)
T_int = 0.5           # Integration time (s)

theta_sat = 0.1       # Maximum angular deflection before saturation (rad)

# ============================================================
# 2. MODEL EQUATIONS
# ============================================================
def H(f):
    w = 2 * np.pi * f
    return 1.0 / (k - I * w**2 + 1j * gamma * w)

def NE(f):
    return NE0 * (1.0 + (f / f_c)**(-alpha))

def noise_denominator(f):
    return p**2 * NE(f) + kappa_L**2 * N_u

def d_prime(As, T, Am=0.0):
    SNR_signal = p**2 * As**2 / 2.0 / noise_denominator(f_s)
    d_linear = np.sqrt(SNR_signal * T)
    if Am == 0.0:
        return d_linear
    delta_f = f_m - f_s
    attenuation = np.sinc(delta_f * T) 
    Lambda_sq = (p**2 * np.abs(H(f_m))**2 * Am**2 * attenuation**2 * (T / 2.0)) / noise_denominator(f_s)
    return d_linear / np.sqrt(1.0 + Lambda_sq)

def As_min(T):
    return np.sqrt(2.0 * noise_denominator(f_s) / (p**2 * T))

def Am_sat():
    return theta_sat / (p * np.abs(H(f_m)))

# ============================================================
# 3. FIGURE 1: TRANSFER FUNCTION
# ============================================================
freqs = np.logspace(0, 3, 500)
H_mag = np.abs(H(freqs))

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(freqs, H_mag, 'k-', lw=1.8)
ax.axvline(f0, color='C0', ls='--', lw=1.2, label=f'Resonance $f_0 = {f0:.0f}$ Hz')
ax.axvspan(1, 50, color='C2', alpha=0.15, label='Biological band (1–50 Hz)')
ax.axvline(f_m, color='C3', ls=':', lw=1.5, label=f'Mains frequency $f_m = {f_m:.0f}$ Hz')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'$|H(2\pi f)|$ (rad / (N$\cdot$m))')
ax.set_title('Deflection transfer function of a mechanosensory hair')
ax.legend(loc='lower left')
ax.grid(True, which='both', alpha=0.3)
fig.savefig('abeja1.png', dpi=300)
fig.savefig('abeja1.pdf')
print("--> Figura 1 guardada como 'abeja1.png' y 'abeja1.pdf'")
plt.show()  # <--- ESTO MUESTRA LA VENTANA EN PANTALLA
plt.close(fig)

# ============================================================
# 4. FIGURE 2: DETECTABILITY CURVES
# ============================================================
As_grid = np.logspace(0, 3.5, 400)

fig, ax = plt.subplots(figsize=(6, 4))
d_curve_clean = np.array([d_prime(a, T_int, Am=0.0) for a in As_grid])
ax.loglog(As_grid, d_curve_clean, 'k-', lw=2.0, label=r'$A_m = 0$ V/m')

d_curve_mains = np.array([d_prime(a, T_int, Am=500.0) for a in As_grid])
ax.loglog(As_grid, d_curve_mains, 'r--', lw=2.0, label=r'$A_m = 500$ V/m (50 Hz)')

ax.axhline(1.0, color='gray', lw=1.0, ls='-.')
ax.text(1.5, 1.1, r"$d' = 1$ (threshold)", color='gray', fontsize=9)
ax.set_xlabel(r'Signal amplitude $A_s$ (V/m)')
ax.set_ylabel(r"Detectability $d'$")
ax.set_title(r'Detectability for $T_{\mathrm{int}} = 0.5$ s: linear interference is rejected')
ax.set_ylim(1e-2, 1e2)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, which='both', alpha=0.3)
fig.savefig('abeja2.png', dpi=300)
fig.savefig('abeja2.pdf')
print("--> Figura 2 guardada como 'abeja2.png' y 'abeja2.pdf'")
plt.show()  # <--- ESTO MUESTRA LA VENTANA EN PANTALLA
plt.close(fig)

# ============================================================
# 5. FIGURE 3: REGIME DIAGRAM (WITH BLINDNESS REGION)
# ============================================================
As_plot = np.logspace(1, 3.5, 200)
Am_grid = np.logspace(-1, 4, 200)
AS, AM = np.meshgrid(As_plot, Am_grid)

D_EFF = np.zeros_like(AS)
for i in range(AS.shape[0]):
    for j in range(AS.shape[1]):
        D_EFF[i, j] = d_prime(AS[i, j], T_int, Am=AM[i, j])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
cs = ax.contourf(AS, AM, np.log10(D_EFF + 1e-6),
                 levels=np.linspace(-2, 2, 21), cmap='RdYlGn', extend='both')
cbar = fig.colorbar(cs, ax=ax, label=r"$\log_{10} d'_{\mathrm{eff}}$")

As_th = As_min(T_int)
ax.axvline(As_th, color='k', lw=1.8, ls='--',
           label=rf'Detection threshold $A_s^{{\min}} = {As_th:.0f}$ V/m')

Am_sat_val = Am_sat()
ax.axhline(Am_sat_val, color='C0', lw=2.0, ls=':',
           label=rf'Saturation threshold $A_m^{{\mathrm{{sat}}}} = {Am_sat_val:.0f}$ V/m')

ax.axhspan(Am_sat_val, Am_grid[-1], color='gray', alpha=0.6, hatch='//')
ax.text(30, Am_sat_val * 2.5, 'Saturation Regime\n(Sensory Blindness)', 
        fontsize=11, fontweight='bold', color='black', ha='left')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Signal amplitude $A_s$ (V/m)')
ax.set_ylabel(r'Mains amplitude $A_m$ (V/m)')
ax.set_title(rf'Regime diagram for $T_{{\mathrm{{int}}}} = {T_int}$ s')
ax.legend(loc='lower right', fontsize=8)
fig.savefig('abeja3.png', dpi=300)
fig.savefig('abeja3.pdf')
print("--> Figura 3 guardada como 'abeja3.png' y 'abeja3.pdf'")
plt.show()  # <--- ESTO MUESTRA LA VENTANA EN PANTALLA
plt.close(fig)

# ============================================================
# 6. SENSITIVITY TABLE
# ============================================================
rows = []
for param, values in [
    ('p',       [0.5e-15, 1.0e-15, 2.0e-15]),
    ('kappa_L', [4.3e-12, 8.6e-12, 1.72e-11]),
    ('f0',      [100.0, 150.0, 250.0]),
    ('Q',       [1.0, 2.0, 4.0]),
    ('NE0',     [1e3, 1e4, 1e5]),
    ('N_u',     [1.35e-5, 1.35e-4, 1.35e-3]),
]:
    for v in values:
        p_s, kL_s, NE0_s, Nu_s, f0_s, Q_s = p, kappa_L, NE0, N_u, f0, Q
        I_s, k_s, gamma_s = I, k, gamma
        
        if param == 'p': p = v
        elif param == 'kappa_L': kappa_L = v
        elif param == 'f0':
            f0 = v; omega0 = 2*np.pi*f0; k = I * omega0**2; gamma = np.sqrt(k*I)/Q
        elif param == 'Q':
            Q = v; gamma = np.sqrt(k*I)/Q
        elif param == 'NE0': NE0 = v
        elif param == 'N_u': N_u = v

        rows.append({
            'Parameter': param,
            'Value': v,
            'A_s_min (V/m)': As_min(T_int),
            'A_m_sat (V/m)': Am_sat()
        })
        
        p, kappa_L, NE0, N_u, f0, Q = p_s, kL_s, NE0_s, Nu_s, f0_s, Q_s
        I, k, gamma = I_s, k_s, gamma_s

df = pd.DataFrame(rows)
df.to_csv('sensitivity_table.csv', index=False)
print("--> Tabla guardada como 'sensitivity_table.csv'\n")

# ============================================================
# 7. PRINT HEADLINE NUMBERS
# ============================================================
print("=== SENSITIVITY TABLE ===")
print(df.to_string(index=False))

print("\n=== HEADLINE NUMBERS FOR LATEX ===")
print(f"Resonance f0 = {f0:.1f} Hz, Q = {Q:.2f}")
print(f"|H(f_s)| at f_s = {f_s} Hz: {np.abs(H(f_s)):.3e} rad/(N·m)")
print(f"|H(f_m)| at f_m = {f_m} Hz: {np.abs(H(f_m)):.3e} rad/(N·m)")
print(f"A_s^min (T_int = {T_int} s): {As_min(T_int):.1f} V/m")
print(f"A_m^sat (theta_sat = {theta_sat} rad): {Am_sat():.1f} V/m")
print(f"Ratio A_s^min / behavioural threshold (300 V/m): {As_min(T_int)/300:.2f}")
print(f"Matched filter attenuation at 50 Hz (T_int=0.5s): {20*np.log10(np.abs(np.sinc((f_m-f_s)*T_int))):.1f} dB")
