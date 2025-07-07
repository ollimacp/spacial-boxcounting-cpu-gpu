import matplotlib.pyplot as plt
import numpy as np


def plot_spatial_results(BoxCountR_map, spa_Lac_map, title='Spatial Box Count Results'):
    """Plot spatial box count and lacunarity maps side by side.

    Parameters:
        BoxCountR_map (np.ndarray): Spatial box count ratio map.
        spa_Lac_map (np.ndarray): Spatial lacunarity map.
        title (str): Title for the entire figure.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im0 = ax[0].imshow(BoxCountR_map, cmap='viridis', interpolation='none')
    ax[0].set_title('Box Count Ratio')
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(spa_Lac_map, cmap='magma', interpolation='none')
    ax[1].set_title('Spatial Lacunarity')
    fig.colorbar(im1, ax=ax[1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_multiscale_regression(BoxSizes, counts, slope, intercept):
    """Plot log-log regression for multi-scale box counting.

    Parameters:
        BoxSizes (list or np.ndarray): List of box sizes used.
        counts (list or np.ndarray): Corresponding box counts.
        slope (float): Slope from regression (fractal dimension is -slope).
        intercept (float): Intercept from regression.
    """
    log_bs = np.log(BoxSizes)
    log_counts = np.log(counts)

    plt.figure(figsize=(8,6))
    plt.scatter(log_bs, log_counts, color='blue', label='Data points')

    # Compute regression line
    line = slope * log_bs + intercept
    plt.plot(log_bs, line, color='red', label=f'Regression Line (slope = {slope:.3f})')
    plt.xlabel('log(Box Size)')
    plt.ylabel('log(Count)')
    plt.title('Multi-scale Box Counting Regression')
    plt.legend()
    plt.show()
