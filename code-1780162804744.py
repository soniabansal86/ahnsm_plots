import numpy as np
import pandas as pd

def f(z): return z**3 - z
def df(z): return 3*z**2 - 1

res = 1000
max_iter = 25 # Lower iteration count to expose slow-converging chaotic regions

x, y = np.meshgrid(np.linspace(-1.5, 1.5, res), np.linspace(-1.5, 1.5, res))
c = x + 1j * y

def get_stats(epsilon):
    z = np.copy(c)
    z_prev = z - 1e-3 - 1e-3j
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
            
            # Increment iteration count for points that haven't converged
            iters_taken += mask_not_converged
            
            z_prev = z
            z = z_next
            
    # Calculate stats
    total_points = res * res
    failed = np.sum(np.abs(f(z)) > 1e-6) # points that didn't converge within max_iter
    avg_iters = np.mean(iters_taken)
    max_iters = np.max(iters_taken)
    
    return {
        'Avg Iterations': round(avg_iters, 2),
        'Divergence / Slow Chaos (%)': round((failed / total_points) * 100, 4)
    }

results = {
    'Pure Newton': get_stats(None),
    'AHNSM (ε = 0.1)': get_stats(0.1),
    'AHNSM (ε = 0.01)': get_stats(0.01),
    'AHNSM (ε = 0.0001)': get_stats(0.0001)
}

df_results = pd.DataFrame(results).T
print(df_results.to_markdown())