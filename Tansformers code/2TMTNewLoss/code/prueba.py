import numpy as np
import matplotlib.pyplot as plt

# Generar datos sintéticos
def generar_datos_sinteticos():
    L_values = [1, 2, 3, 4]
    Cn2_values = [1e-14-, 3e-14-, 5e-14-, 7e-14-,9e-14-,1e-15-, 3e-15-, 5e-15-, 7e-15-,9e-15-]
    np.random.seed(42)

    psnr_dict = {}
    ssim_dict = {}

    for L in L_values:
        for Cn2 in Cn2_values:
            psnr_values = np.random.uniform(20, 40, size=5).tolist()
            ssim_values = np.random.uniform(0.7, 0.95, size=5).tolist()
            
            psnr_dict[(L, Cn2)] = psnr_values
            ssim_dict[(L, Cn2)] = ssim_values

    return psnr_dict, ssim_dict

# Función para calcular promedios
def calcular_promedios(metrics_dict):
    promedios = {}
    for key, values in metrics_dict.items():
        promedios[key] = np.mean(values)
    return promedios

# Función para graficar resultados
def graficar_resultados(metric1_dict, metric2_dict, title1="PSNR", title2="SSIM"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for metric_dict, title, axis in zip([metric1_dict, metric2_dict], [title1, title2], ax):
        L_values = sorted(set(key[0] for key in metric_dict.keys()))
        Cn2_values = sorted(set(key[1] for key in metric_dict.keys()))

        for L in L_values:
            y_values = [np.mean(metric_dict[(L, Cn2)]) for Cn2 in Cn2_values]
            axis.plot(Cn2_values, y_values, label=f'L = {L}')

        axis.set_title(title)
        axis.set_xlabel('Cn2')
        axis.set_ylabel('Metric Value')
        axis.legend()
        axis.grid(True)

    plt.tight_layout()
    plt.show()

# Uso del código
psnr_dict, ssim_dict = generar_datos_sinteticos()

psnr_promedios = calcular_promedios(psnr_dict)
ssim_promedios = calcular_promedios(ssim_dict)

print("Promedios PSNR:", psnr_promedios)
print("Promedios SSIM:", ssim_promedios)

graficar_resultados(psnr_dict, ssim_dict)
