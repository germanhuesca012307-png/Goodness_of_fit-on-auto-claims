import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt

# read csv
df = pd.read_csv('insurance.csv')
filtro = df['incident_type'].isin(['Single Vehicle Collision', 'Multi-vehicle Collision'])
montos = df.loc[filtro,'total_claim_amount'].to_numpy()
n = len(montos)
x_barra = np.mean(montos)

# inicial value for alpha
alpha = (x_barra**2) / np.var(montos, ddof=1)
beta = 0
max_iter = 100
tol = 1e-6

#nwt-R 

C = np.log(x_barra) - (1/n) * np.sum(np.log(montos))

def f0(a=alpha) :
    return np.log(a) - sp.digamma(a) - C 
def f1(a=alpha) :
    return a**(-1) - sp.polygamma(1,a) 
def f2(a=alpha) :
    return -a**(-2) - sp.polygamma(2,a)

def paso_newton(a):
    val_f0 = f0(a)
    val_f1 = f1(a)
    val_f2 = f2(a)
    
    num = val_f0 * val_f1
    den = (val_f1**2) - (val_f0 * val_f2)
    return a - (num / den)

for i in range(max_iter):
    alpha_nuevo = paso_newton(alpha)
    error = abs(1 - alpha/alpha_nuevo)
    
    alpha = alpha_nuevo
    if error < tol:
        print(f"Convergió en la iteración {i+1}")
        break

# find beta given alpha
beta = x_barra / alpha

print(f"alpha: {alpha:.2f}")
print(f"beta : {beta:.2f}")

plt.figure(figsize = (9,5))

#hist a densidad
plt.hist(
    montos,
    bins=round(np.sqrt(len(montos))),
    density=True,
    color='lightgreen',
    edgecolor='black',
    alpha=.6,
    label= 'histograma de montos de reclamacion'
)

#gammas' pdf 
def pdf_gamma(x,a=alpha,b=beta) :
    return (x**(a-1)*np.exp(-x/b))/((b**a)*sp.gamma(a)) 

x_curva = np.linspace(min(montos),max(montos))
y_curva = pdf_gamma(x_curva,alpha,beta)

plt.plot(x_curva,y_curva,color='red', linewidth=2, label=f'Ajuste Gamma (α={alpha:.2f}, θ={beta:.1f})')
plt.title('Ajuste de Distribución Gamma a Montos de Siniestros')
plt.xlabel('Monto de Reclamación ($)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

#goodness of fit

def trapecios(f, a, b, n):
    if a == b: 
        return 0.0
    h = (b - a) / n
    suma_interna = sum(f(a + i * h) for i in range(1, n)) 
    resultado = (f(a) + 2 * suma_interna + f(b)) * h / 2 
    return resultado

def trapecio_iterativo_ks(f, datos, n_trapecios=100):
    x_ord = np.sort(datos)
    n_datos = len(x_ord)
    
    F_acumulada = 0.0
    diferencias = []
    x_anterior = 1e-6 if alpha < 1 else 0.0 
    
    for i in range(n_datos):
        x_actual = x_ord[i]   
        pedacito = trapecios(f, x_anterior, x_actual, n_trapecios)
        F_acumulada = F_acumulada + pedacito
        Fn = (i + 1) / n_datos 
        dif = abs(Fn - F_acumulada)
        diferencias.append(dif)
        x_anterior = x_actual
        
    D = max(diferencias)
    return D, diferencias, x_ord

#sacamos el estadístico D
N_subintervalos = 50 
estadistico_D, lista_dif, montos_ordenados = trapecio_iterativo_ks(pdf_gamma, montos, N_subintervalos)
valor_critico = 1.36 / np.sqrt(n)

print(f"Estadístico D : {estadistico_D:.6f}")
print(f"Valor crítico : {valor_critico:.6f}")

if estadistico_D < valor_critico:
    print("Conclusión: No se rechaza H0 (El ajuste Gamma es bueno)")
else:
    print("Conclusión: Se rechaza H0 (Los datos no siguen esta Gamma)")

plt.show()

