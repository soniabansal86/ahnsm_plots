import numpy as np

def f(z): return z**3 - z
def df(z): return 3*z**2 - 1

roots = np.array([0, 1, -1])
res = 1000
max_iter = 40

x, y = np.meshgrid(np.linspace(-1.5, 1.5, res), np.linspace(-1.5, 1.5, res))
c = x + 1j * y

def simulate_ahnsm(epsilon):
    z = np.copy(c)
    z_prev = z - 1e-3 - 1e-3j
    
    for _ in range(max_iter):
        fz = f(z)
        dfz = df(z)
        deriv_mag = np.abs(dfz)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            step_n = fz / dfz
            fz_prev = f(z_prev)
            den = fz - fz_prev
            den = np.where(den == 0, 1e-15, den)
            step_s = fz * (z - z_prev) / den
            
            # If epsilon is None, do Pure Newton
            if epsilon is None:
                step = step_n
            else:
                step = np.where(deriv_mag > epsilon, step_n, step_s)
            
            mask_not_converged = np.abs(fz) > 1e-6
            z_next = np.where(mask_not_converged, z - step, z)
            
            z_prev = z
            z = z_next
            
    # Calculate stats
    converged_to_0 = np.sum(np.abs(z - 0) < 1e-2)
    converged_to_1 = np.sum(np.abs(z - 1) < 1e-2)
    converged_to_m1 = np.sum(np.abs(z - -1) < 1e-2)
    
    total_converged = converged_to_0 + converged_to_1 + converged_to_m1
    total_points = res * res
    failed = total_points - total_converged
    
    return {
        'root_0': (converged_to_0 / total_points) * 100,
        'root_1': (converged_to_1 / total_points) * 100,
        'root_m1': (converged_to_m1 / total_points) * 100,
        'failed': (failed / total_points) * 100
    }

results = {
    'Pure Newton': simulate_ahnsm(None),
    'AHNSM (eps=0.1)': simulate_ahnsm(0.1),
    'AHNSM (eps=0.01)': simulate_ahnsm(0.01),
    'AHNSM (eps=0.0001)': simulate_ahnsm(0.0001)
}

import pandas as pd
df_results = pd.DataFrame(results).T
print(df_results.to_markdown())