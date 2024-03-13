import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tab = np.load("noisyNet_test/results/plot_noisytrue_set_sizes.npy")
span = 10  # Ajustez cette valeur pour modifier la "lissité" de l'EMA

plt.figure(figsize=(10, 6))
i=0
for set_sizes in range(5, 130, 8):
    
    print(set_sizes, " -> ", i)
    i+=1
# Calculez et tracez l'EMA pour chaque série dans tab
for i in range(len(tab)):
    if i < 8:
        continue
    # Convertissez la série en DataFrame pandas
    series = pd.Series(tab[i])
    
    # Calculez l'EMA
    ema = series.ewm(span=span, adjust=False).mean()
    
    # Tracez l'EMA
    plt.plot(ema, label=f"EMA {i}")

plt.legend()
plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from ipywidgets import interact, widgets

# # Simulons le chargement de votre tableau
# # tab = np.load("noisyNet_test/results/plot_noisytrue_set_sizes.npy")
# tab = np.random.rand(5, 100)  # Utilisé pour la simulation, à remplacer par votre ligne de chargement

# def plot_data(include_plots):
#     plt.figure(figsize=(10, 6))
#     for i in include_plots:
#         plt.plot(tab[i], label=f"Plot {i}")
#     plt.legend()
#     plt.show()

# include_plots = widgets.SelectMultiple(
#     options=range(len(tab)),
#     value=(0,),  # les indices des plots à afficher par défaut
#     description='Plots',
#     disabled=False
# )

# interact(plot_data, include_plots=include_plots)
