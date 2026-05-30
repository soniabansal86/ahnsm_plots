import numpy as np
import pandas as pd

def f(z): return z**3 - z
def df(z): return 3*z**2 - 1

res = 500
x, y = np.meshgrid(np.linspace(-1.5, 1.5, res), np.linspace(-1.5, 1.5, res))
c = x + 1j * y

def get_step_ratio(epsilon):
    z = np.copy(c)
    z_prev = z - 1e-3 - 1e-3j
    
    total_newton = np.zeros(z.shape)
    total_secant = np.zeros(z.shape)
    
    for i in range(15):
        fz = f(z)
        dfz = df(z)
        deriv_mag = np.abs(dfz)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            step_n = fz / dfz
            fz_prev = f(z_prev)
            den = fz - fz_prev
            den = np.where(den == 0, 1e-15, den)
            step_s = fz * (z - z_prev) / den
            
            mask_not_converged = np.abs(fz) > 1e-6
            
            if epsilon is None:
                step = step_n
                total_newton += mask_not_converged
            else:
                is_newton = deriv_mag > epsilon
                step = np.where(is_newton, step_n, step_s)
                total_newton += mask_not_converged & is_newton
                total_secant += mask_not_converged & ~is_newton
            
            z_next = np.where(mask_not_converged, z - step, z)
            z_prev = z
            z = z_next
            
    n_sum = np.sum(total_newton)
    s_sum = np.sum(total_secant)
    total = n_sum + s_sum
    
    return {
        'Newton Steps (%)': round((n_sum / total) * 100, 2),
        'Secant Steps (%)': round((s_sum / total) * 100, 2)
    }

results = {
    'Pure Newton': get_step_ratio(None),
    'AHNSM (ε = 10^-1)': get_step_ratio(0.1),
    'AHNSM (ε = 10^-2)': get_step_ratio(0.01),
    'AHNSM (ε = 10^-4)': get_step_ratio(0.0001)
}

df_results = pd.DataFrame(results).T
print(df_results.to_markdown())