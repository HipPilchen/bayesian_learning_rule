import numpy as np

def gaussian_density_1D(x, mu, sigma_2):
    """
    Evaluate the density of a Gaussian distribution at a point x.
    
    Parameters:
    - x: The point at which to evaluate the density
    - mu: The mean of the Gaussian distribution
    - sigma: The standard deviation of the Gaussian distribution
    
    Returns:
    - The density of the Gaussian distribution at point x
    """
    return (1 / (  np.sqrt(2 * np.pi * sigma_2))) * np.exp( -0.5 * ((x - mu)**2 / sigma_2) )

from PIL import Image
import io
import matplotlib.pyplot as plt

# Redéfinition de la fonction pour utiliser Pillow pour créer le GIF
def create_gif_function_eval_pillow(tab_x, f, filename="function_eval_pillow.gif"):
    images = []
    x_min, x_max = min(tab_x), max(tab_x)
    y_values = [f(x) for x in tab_x]
    print(y_values)
    y_min, y_max = min(y_values), max(y_values)

    for i in range(1, len(tab_x) + 1):
        fig, ax = plt.subplots()
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max)))
        ax.plot(tab_x[:i], [f(x) for x in tab_x[:i]], 'r')
        
        # Sauvegarder la figure dans un buffer (en mémoire), puis charger avec Pillow
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        images.append(img)
        plt.close(fig)
    
    # Sauvegarder toutes les images dans un GIF
    images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

# Appeler la fonction avec le tableau tab_x exemple et la fonction f
# Redéfinition de la fonction pour inclure la fonction de base en bleu
def create_gif_function_eval_pillow_with_base(tab_x, f, step_size = 1, filename="function_eval_base_pillow.gif"):
    images = []
    x_min, x_max = min(tab_x), max(tab_x)
    y_values = [f(x) for x in tab_x]
    y_min, y_max = min(y_values), max(y_values)

    # Tracé de la fonction complète en bleu pour utilisation comme fond
    fig, ax = plt.subplots()
    ax.plot(tab_x, y_values, 'b', alpha=0.3)  # Utiliser une transparence pour distinguer la progression
    base_img_buf = io.BytesIO()
    plt.savefig(base_img_buf, format='png')
    base_img_buf.seek(0)
    base_image = Image.open(base_img_buf)
    plt.close(fig)
    
    for i in range(1, len(tab_x) + 1, step_size):
        fig, ax = plt.subplots()
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max)))
        
        # Tracer d'abord la fonction de base en bleu
        ax.plot(tab_x, y_values, 'b', alpha=0.3)  # Même tracé que le fond
        # Puis, superposer la progression en rouge
        ax.plot(tab_x[:i], [f(x) for x in tab_x[:i]], 'r')
        
        # Sauvegarder cette étape comme une image PIL
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        images.append(img)
        plt.close(fig)
    
    # Sauvegarder toutes les images dans un GIF, avec la fonction de base en fond
    images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

if __name__ == "__main__":
    # Tester la fonction avec un tableau tab_x exemple
    tab_x = np.linspace(-np.pi, np.pi, 100)
    create_gif_function_eval_pillow_with_base(tab_x, np.sin,step_size = 10, filename="sin.gif")
