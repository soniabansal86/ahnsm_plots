import numpy as np
import pandas as pd

def f(z): return z**3 - z
def df(z): return 3*z**2 - 1

# Only look at the danger zone near the critical point ~0.577
res = 1000
max_iter = 50 
x, y = np.meshgrid(np.linspace(0.50, 0.65, res), np.linspace(-0.5, 0.5, res))
c = x + 1j * y

def get_danger_zone_stats(epsilon):
    z = np.copy(c)
    z_prev = z - 1e-4 - 1e-4j
    iters_taken = np.zeros(z.shape)
    
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
            
            iters_taken += mask_not_converged
            
            z_prev = z
            z = z_next
            
    return {
        'Average Iterations': round(np.mean(iters_taken), 2),
        'Max Iterations': int(np.max(iters_taken)),
        'Failed to Converge (%)': round(np.mean(iters_taken == max_iter) * 100, 2)
    }

results = {
    'Pure Newton': get_danger_zone_stats(None),
    'AHNSM (ε = 10^-1)': get_danger_zone_stats(0.1),
    'AHNSM (ε = 10^-2)': get_danger_zone_stats(0.01),
    'AHNSM (ε = 10^-4)': get_danger_zone_stats(0.0001)
}

df_results = pd.DataFrame(results).T
print(df_results.to_markdown())