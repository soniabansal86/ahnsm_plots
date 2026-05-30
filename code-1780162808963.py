import numpy as np
import pandas as pd

def f(z): return z**3 - z
def df(z): return 3*z**2 - 1

res = 1000
max_iter = 10 

x, y = np.meshgrid(np.linspace(-1.5, 1.5, res), np.linspace(-1.5, 1.5, res))
c = x + 1j * y

def get_max_step(epsilon):
    z = np.copy(c)
    z_prev = z - 1e-3 - 1e-3j
    max_step_magnitude = 0
    
    for i in range(max_iter):
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
            
            # Record max step taken by any point that hasn't converged yet
            current_steps = np.abs(step)
            current_steps = np.where(mask_not_converged, current_steps, 0)
            
            # Handle NaNs and infs safely for max calculation
            current_steps = np.nan_to_num(current_steps, posinf=0)
            
            iter_max = np.max(current_steps)
            if iter_max > max_step_magnitude:
                max_step_magnitude = iter_max
            
            z_next = np.where(mask_not_converged, z - step, z)
            z_prev = z
            z = z_next
            
    return round(max_step_magnitude, 2)

results = {
    'Pure Newton': get_max_step(None),
    'AHNSM (ε = 10^-1)': get_max_step(0.1),
    'AHNSM (ε = 10^-2)': get_max_step(0.01),
    'AHNSM (ε = 10^-4)': get_max_step(0.0001)
}

df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Maximum Computational Step Size (|z_{n+1} - z_n|)'])
print(df_results.to_markdown())