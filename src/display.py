"""
Module for displaying forecast comparisons using Prophet model.
Updated: 2026-29-06

"""


import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict

def plot_forecast_comparison(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    kpi: str,
    figsize: tuple = (12, 5),
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot train/test actual vs predicted values from Prophet model.
    
    Parameters:
    -----------
    model : Prophet
        Fitted Prophet model object
    train_data : pd.DataFrame
        Training data with columns ['ds', 'y']
    test_data : pd.DataFrame
        Test data with columns ['ds', 'y']
    kpi : str
        KPI name for labels and title
    figsize : tuple, default=(12, 5)
        Figure size (width, height)
    colors : dict, optional
        Custom colors for plot lines
        Keys: 'train_actual', 'test_actual', 'train_pred', 'test_pred'
    save_path : str, optional
        Path to save the plot (e.g., 'forecast.png')
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    None (displays plot)
    
    Dependencies:
    ------------
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import Optional, Dict
    
    Example:
    --------
    >>> plot_forecast_comparison(
    ...     model=prophet_model,
    ...     train_data=train_df,
    ...     test_data=test_df,
    ...     kpi='Sales',
    ...     figsize=(14, 6)
    ... )
    """

    
    # Default colors
    default_colors = {
        'train_actual': 'tab:blue',
        'test_actual': 'tab:green',
        'train_pred': 'tab:orange',
        'test_pred': 'tab:red'
    }
    
    # Update with custom colors if provided
    if colors:
        default_colors.update(colors)

    # Detect Prophet extra regressors (works for univariate and multivariate)
    extra_regressors = list(getattr(model, "extra_regressors", {}).keys())

    # Build future dataframe for full timeline (train + test)
    full_data = pd.concat([train_data.copy(), test_data.copy()], axis=0, ignore_index=True, sort=False)

    # Validate required columns
    required_cols = ["ds"] + extra_regressors
    missing_cols = [c for c in required_cols if c not in full_data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for prediction: {missing_cols}. "
            f"Expected columns: {required_cols}"
        )

    full_future = full_data[required_cols].copy()

    # Fill missing regressor values for robust prediction
    for reg in extra_regressors:
        fill_val = train_data[reg].median() if reg in train_data.columns else full_future[reg].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        full_future[reg] = full_future[reg].fillna(fill_val)

    forecast_full = model.predict(full_future)

    # Build actual frames
    train_actual = train_data.rename(columns={"y": "actual"}).copy()
    test_actual = test_data.rename(columns={"y": "actual"}).copy()

    # Build predicted frames
    pred = forecast_full[["ds", "yhat"]].copy()
    train_pred = train_actual[["ds"]].merge(pred, on="ds", how="left")
    test_pred = test_actual[["ds"]].merge(pred, on="ds", how="left")

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(
        train_actual["ds"],
        train_actual["actual"],
        color=default_colors['train_actual'],
        label="Train Actual",
        linewidth=2
    )
    plt.plot(
        test_actual["ds"],
        test_actual["actual"],
        color=default_colors['test_actual'],
        label="Test Actual",
        linewidth=2
    )
    plt.plot(
        train_pred["ds"],
        train_pred["yhat"],
        color=default_colors['train_pred'],
        linestyle="--",
        label="Train Predicted",
        linewidth=2
    )
    plt.plot(
        test_pred["ds"],
        test_pred["yhat"],
        color=default_colors['test_pred'],
        linestyle="--",
        label="Test Predicted",
        linewidth=2
    )

    plt.xlabel("Time", fontsize=11)
    plt.ylabel(kpi, fontsize=11)
    plt.title(f"Train/Test Actual vs Predicted ({kpi})", fontsize=12, fontweight='bold')
    plt.suptitle("Model Name: Prophet", y=1.02, fontsize=10)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
        
        
if __name__ == "__main__":
    pass