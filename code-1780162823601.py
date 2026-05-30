import numpy as np
import pandas as pd

def f(z): return z**3 - z
def df(z): return 3*z**2 - 1

res = 2000
max_iter = 15 # strict limit to penalize chaotic bounding

x, y = np.meshgrid(np.linspace(-1.5, 1.5, res), np.linspace(-1.5, 1.5, res))
c = x + 1j * y

def get_basin_measure(epsilon):
    z = np.copy(c)
    # Give AHNSM a fair start by making the first step purely Newton if we aren't at a singularity
    # But to keep it simple, just initialize z_prev via a small Newton step
    with np.errstate(divide='ignore', invalid='ignore'):
        z_prev = z - (f(z)/df(z)) * 0.01 
        # fallback if derivative is exactly 0
        z_prev = np.where(np.isnan(z_prev), z - 1e-4, z_prev)
    
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
            z_next = np.where(mask_not_converged, z - step, z)
            
            z_prev = z
            z = z_next
            
    # Calculate what percentage of points successfully converged to one of the 3 roots
    converged = np.sum(np.abs(f(z)) < 1e-3)
    total_points = res * res
    
    return round((converged / total_points) * 100, 2)

results = {
    'Pure Newton': get_basin_measure(None),
    'AHNSM (ε = 10^-1)': get_basin_measure(0.1),
    'AHNSM (ε = 10^-2)': get_basin_measure(0.01),
    'AHNSM (ε = 10^-4)': get_basin_measure(0.0001)
}

df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Global Convergence Area (%)'])
print(df_results.to_markdown())