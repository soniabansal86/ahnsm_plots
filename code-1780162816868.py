import numpy as np
import pandas as pd

def f(z): return z**3 - z
def df(z): return 3*z**2 - 1

# Zoom in directly on the singularity to catch the explosion
res = 2000
x, y = np.meshgrid(np.linspace(0.57, 0.58, res), np.linspace(-0.01, 0.01, res))
c = x + 1j * y

def get_max_step(epsilon):
    z = np.copy(c)
    z_prev = z - 1e-3 - 1e-3j
    max_step_magnitude = 0
    
    for i in range(5): # We only need a few iterations to see the explosion
        fz = f(z)
        dfz = df(z)
        deriv_mag = np.abs(dfz)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            step_n = fz / dfz
            fz_prev = f(z_prev)
            den = fz - fz_prev
            den = np.where(den == 0, 1e-15, den)
            step_s = fz * (z - z_prev) / den
            
            if epsilon is None: step = step_n
            else: step = np.where(deriv_mag > epsilon, step_n, step_s)
            
            mask_not_converged = np.abs(fz) > 1e-6
            
            current_steps = np.abs(step)
            current_steps = np.where(mask_not_converged, current_steps, 0)
            current_steps = np.nan_to_num(current_steps, posinf=0)
            
            iter_max = np.max(current_steps)
            if iter_max > max_step_magnitude:
                max_step_magnitude = iter_max
            
            z_next = np.where(mask_not_converged, z - step, z)
            z_prev = z
            z = z_next
            
    return max_step_magnitude

results = {
    'Pure Newton': get_max_step(None),
    'AHNSM (ε = 10^-1)': get_max_step(0.1),
    'AHNSM (ε = 10^-2)': get_max_step(0.01),
    'AHNSM (ε = 10^-3)': get_max_step(0.001),
    'AHNSM (ε = 10^-4)': get_max_step(0.0001)
}

# format in scientific notation for huge numbers
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Maximum Step Magnitude (|z_n+1 - z_n|)'])
print(df_results.to_markdown())