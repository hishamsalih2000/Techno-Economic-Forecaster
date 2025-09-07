# src/utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import config

def plot_results(y_true, y_pred, target_column, model_save_path, r2_score, mae_score, unit_name):
    """
    Generates and saves a 'Predicted vs. Actual' scatter plot
    with performance metrics displayed on the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label='Predictions')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    
    stats_text = (
        f"R² = {r2_score:.4f}\n"
        f"MAE = {mae_score:.2f} {unit_name}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
            
    ax.set_xlabel(f"Actual {target_column} ({unit_name})", fontsize=14)
    ax.set_ylabel(f"Predicted {target_column} ({unit_name})", fontsize=14)
    ax.set_title(f"Model Performance: Actual vs. Predicted Values", fontsize=16)
    ax.legend()
    
    plot_path = config.IMAGES_DIR / f"{pathlib.Path(model_save_path).stem}_performance_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"High-quality performance plot saved to: {plot_path}")
    plt.close()