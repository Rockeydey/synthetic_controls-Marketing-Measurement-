"""
Updated: 2026-06-30

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings

@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual analysis"""
    campaign_regressors: List[str]
    test_period_col: str = "ds"
    kpi_col: str = "y"
    counterfactual_value: float = 0.0
    include_confidence_interval: bool = True
    ci_alpha: float = 0.15
    aggregate_level: str = "total"  # 'total', 'monthly', 'weekly'
    plot_enabled: bool = True
    plot_figsize: Tuple[int, int] = (12, 5)
    plot_linewidth: float = 2.0
    verbose: bool = True


class CounterfactualImpactAnalyzer:
    """
    Counterfactual Impact Analyzer using Prophet models.
    
    Computes the incremental impact of campaign regressors by comparing
    actual outcomes against a "no campaign" counterfactual scenario.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config (Dict): Configuration dictionary with the following keys:
                - 'campaign_regressors' (list, required): Regressors to zero out
                - 'test_period_col' (str, default='ds'): Date column name
                - 'kpi_col' (str, default='y'): KPI column name
                - 'counterfactual_value' (float, default=0.0): Value to set regressors to
                - 'include_confidence_interval' (bool, default=True): Include CI bands
                - 'ci_alpha' (float, default=0.15): Confidence interval transparency
                - 'plot_enabled' (bool, default=True): Generate plots
                - 'verbose' (bool, default=True): Print logs
        """
        self.config = self._init_config(config or {})
        self.model = None
        self.test_data = None
        self.all_regressors = []
        self.valid_campaign_regressors = []
        self.counterfactual_forecast = None
        self.impact_results = None
        self.impact_summary = None
    
    def _init_config(self, config_dict: Dict[str, Any]) -> CounterfactualConfig:
        """Initialize and validate configuration"""
        if 'campaign_regressors' not in config_dict:
            raise ValueError("'campaign_regressors' is required in configuration")
        
        return CounterfactualConfig(
            campaign_regressors=config_dict['campaign_regressors'],
            test_period_col=config_dict.get('test_period_col', 'ds'),
            kpi_col=config_dict.get('kpi_col', 'y'),
            counterfactual_value=config_dict.get('counterfactual_value', 0.0),
            include_confidence_interval=config_dict.get('include_confidence_interval', True),
            ci_alpha=config_dict.get('ci_alpha', 0.15),
            aggregate_level=config_dict.get('aggregate_level', 'total'),
            plot_enabled=config_dict.get('plot_enabled', True),
            plot_figsize=config_dict.get('plot_figsize', (12, 5)),
            plot_linewidth=config_dict.get('plot_linewidth', 2.0),
            verbose=config_dict.get('verbose', True)
        )
    
    def _log(self, message: str) -> None:
        """Print log message if verbose enabled"""
        if self.config.verbose:
            print(f"[INFO] {message}")
    
    def analyze(self, 
                model,
                test_data: pd.DataFrame,
                model_regressors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete counterfactual impact analysis.
        
        Args:
            model: Fitted Prophet model
            test_data (pd.DataFrame): Test dataset with regressors and KPI
            model_regressors (list): List of all regressors used in model
            
        Returns:
            Dict: Complete analysis results including forecasts, metrics, summary
        """
        self._log("Starting counterfactual impact analysis...")
        
        # Validate inputs
        self._validate_inputs(model, test_data)
        
        # Store inputs
        self.model = model
        self.test_data = test_data.copy()
        
        # Determine regressors
        if model_regressors:
            self.all_regressors = model_regressors
        else:
            self.all_regressors = list(model.extra_regressors.keys())
        
        if not self.all_regressors:
            raise ValueError("No regressors found in model. Cannot run counterfactual analysis.")
        
        # Validate and filter campaign regressors
        self._validate_campaign_regressors()
        
        # Generate counterfactual forecasts
        self._generate_counterfactual_forecast()
        
        # Compute impact metrics
        self._compute_impact()
        
        # Generate summary
        self._generate_summary()
        
        self._log("Counterfactual impact analysis completed successfully.")
        
        return self.get_results()
    
    def _validate_inputs(self, model, test_data: pd.DataFrame) -> None:
        """Validate input model and data"""
        if model is None:
            raise ValueError("Model cannot be None")
        
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("test_data must be a pandas DataFrame")
        
        if len(test_data) == 0:
            raise ValueError("test_data cannot be empty")
        
        if self.config.test_period_col not in test_data.columns:
            raise ValueError(f"Column '{self.config.test_period_col}' not found in test_data")
        
        if self.config.kpi_col not in test_data.columns:
            raise ValueError(f"Column '{self.config.kpi_col}' not found in test_data")
    
    def _validate_campaign_regressors(self) -> None:
        """Validate and filter campaign regressors"""
        # Check regressors exist in model
        invalid_model_regs = [r for r in self.config.campaign_regressors if r not in self.all_regressors]
        if invalid_model_regs:
            warnings.warn(f"Regressors not in model: {invalid_model_regs}. They will be ignored.")
        
        # Check regressors exist in test data
        invalid_data_regs = [r for r in self.config.campaign_regressors if r not in self.test_data.columns]
        if invalid_data_regs:
            warnings.warn(f"Regressors not in test_data: {invalid_data_regs}. They will be ignored.")
        
        # Keep only valid campaign regressors
        self.valid_campaign_regressors = [
            r for r in self.config.campaign_regressors 
            if r in self.all_regressors and r in self.test_data.columns
        ]
        
        if not self.valid_campaign_regressors:
            raise ValueError(
                f"No valid campaign regressors found. "
                f"Config: {self.config.campaign_regressors}, "
                f"Model: {self.all_regressors}, "
                f"Data: {self.test_data.columns.tolist()}"
            )
        
        self._log(f"Campaign regressors to zero out: {self.valid_campaign_regressors}")
    
    def _generate_counterfactual_forecast(self) -> None:
        """Generate counterfactual forecast with campaign regressors set to 0"""
        self._log("Generating counterfactual forecast...")
        
        # Create counterfactual input: all regressors as observed
        future_cf = self.test_data[[self.config.test_period_col] + self.all_regressors].copy()
        
        # Zero out campaign regressors
        for regressor in self.valid_campaign_regressors:
            future_cf[regressor] = self.config.counterfactual_value
        
        # Generate forecast
        forecast_cf = self.model.predict(future_cf)
        
        # Rename columns
        self.counterfactual_forecast = forecast_cf[[
            self.config.test_period_col, "yhat", "yhat_lower", "yhat_upper"
        ]].rename(columns={
            "yhat": "yhat_cf",
            "yhat_lower": "yhat_cf_lower",
            "yhat_upper": "yhat_cf_upper"
        })
        
        self._log(f"Counterfactual forecast generated: {len(self.counterfactual_forecast)} periods")
    
    def _compute_impact(self) -> None:
        """Compute incremental impact of campaigns"""
        self._log("Computing incremental impact...")
        
        # Join actual and counterfactual
        self.impact_results = self.test_data[[
            self.config.test_period_col, self.config.kpi_col
        ]].merge(
            self.counterfactual_forecast,
            on=self.config.test_period_col,
            how="left"
        )
        
        # Compute impact metrics
        self.impact_results["incremental_impact"] = (
            self.impact_results[self.config.kpi_col] - 
            self.impact_results["yhat_cf"]
        )
        
        # Compute percentage impact
        self.impact_results["incremental_impact_pct"] = np.where(
            self.impact_results["yhat_cf"] != 0,
            (self.impact_results["incremental_impact"] / self.impact_results["yhat_cf"]) * 100,
            np.nan
        )
        
        self._log("Impact metrics computed")
    
    def _generate_summary(self) -> None:
        """Generate summary statistics"""
        self._log("Generating impact summary...")
        
        total_actual = float(self.impact_results[self.config.kpi_col].sum())
        total_cf = float(self.impact_results["yhat_cf"].sum())
        total_impact = float(self.impact_results["incremental_impact"].sum())
        
        # Avoid division by zero
        if total_cf != 0:
            lift_pct = (total_impact / total_cf) * 100
        else:
            lift_pct = np.nan
        
        self.impact_summary = {
            "campaign_regressors": self.valid_campaign_regressors,
            "num_campaigns": len(self.valid_campaign_regressors),
            "test_period_start": self.impact_results[self.config.test_period_col].min(),
            "test_period_end": self.impact_results[self.config.test_period_col].max(),
            "num_periods": len(self.impact_results),
            "total_actual_kpi": total_actual,
            "total_counterfactual_kpi": total_cf,
            "total_incremental_impact": total_impact,
            "incremental_lift_pct": lift_pct,
            "avg_period_incremental_impact": float(self.impact_results["incremental_impact"].mean()),
            "median_period_incremental_impact": float(self.impact_results["incremental_impact"].median()),
            "min_period_incremental_impact": float(self.impact_results["incremental_impact"].min()),
            "max_period_incremental_impact": float(self.impact_results["incremental_impact"].max()),
            "std_incremental_impact": float(self.impact_results["incremental_impact"].std()),
            "periods_positive_impact": int((self.impact_results["incremental_impact"] > 0).sum()),
            "periods_negative_impact": int((self.impact_results["incremental_impact"] < 0).sum()),
        }
        
        self._log("Summary statistics generated")
    
    def get_results(self) -> Dict[str, Any]:
        """Return complete analysis results"""
        return {
            "impact_df": self.impact_results,
            "summary": self.impact_summary,
            "config": self.config,
            "counterfactual_forecast": self.counterfactual_forecast,
            "valid_campaign_regressors": self.valid_campaign_regressors
        }
    
    def print_summary(self) -> None:
        """Print formatted summary statistics"""
        if self.impact_summary is None:
            print("[ERROR] Analysis not run yet. Call analyze() first.")
            return
        
        print("\n" + "="*70)
        print("COUNTERFACTUAL IMPACT ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nCampaign Regressors ({len(self.valid_campaign_regressors)}):")
        for i, reg in enumerate(self.valid_campaign_regressors, 1):
            print(f"  {i}. {reg}")
        
        print(f"\nTest Period:")
        print(f"  Start: {self.impact_summary['test_period_start']}")
        print(f"  End: {self.impact_summary['test_period_end']}")
        print(f"  Periods: {self.impact_summary['num_periods']}")
        
        print(f"\nKPI Summary:")
        print(f"  Total Actual KPI: {self.impact_summary['total_actual_kpi']:,.2f}")
        print(f"  Total Counterfactual (No Campaign): {self.impact_summary['total_counterfactual_kpi']:,.2f}")
        print(f"  Total Incremental Impact: {self.impact_summary['total_incremental_impact']:,.2f}")
        print(f"  Incremental Lift %: {self.impact_summary['incremental_lift_pct']:.2f}%")
        
        print(f"\nPeriod-Level Impact:")
        print(f"  Average: {self.impact_summary['avg_period_incremental_impact']:,.2f}")
        print(f"  Median: {self.impact_summary['median_period_incremental_impact']:,.2f}")
        print(f"  Min: {self.impact_summary['min_period_incremental_impact']:,.2f}")
        print(f"  Max: {self.impact_summary['max_period_incremental_impact']:,.2f}")
        print(f"  Std Dev: {self.impact_summary['std_incremental_impact']:,.2f}")
        
        print(f"\nImpact Direction:")
        print(f"  Periods with Positive Impact: {self.impact_summary['periods_positive_impact']}")
        print(f"  Periods with Negative Impact: {self.impact_summary['periods_negative_impact']}")
        
        print("\n" + "="*70 + "\n")
    
    def _resolve_png_path(self, save_path: Optional[str], default_filename: str) -> str:
        """Return a valid .png path (force .png extension)."""
        path = save_path or default_filename
        root, ext = os.path.splitext(path)

        if ext.lower() != ".png":
            path = f"{root}.png" if ext else f"{path}.png"

        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        return path
    
    def plot_impact(self, 
                    save_path: Optional[str] = None,
                    title: Optional[str] = None,
                    save_png: bool = False) -> None:
        """
        Plot actual vs counterfactual KPI with confidence intervals.
        
        Args:
            save_path (str): Optional path to save the plot (forced to .png)
            title (str): Optional custom title
            save_png (bool): If True, save plot as .png using default name when save_path is None
        """
        if self.impact_results is None:
            print("[ERROR] Analysis not run yet. Call analyze() first.")
            return
        
        if not self.config.plot_enabled:
            self._log("Plotting disabled in config")
            return
        
        plt.figure(figsize=self.config.plot_figsize)
        
        # Plot actual KPI
        plt.plot(
            self.impact_results[self.config.test_period_col],
            self.impact_results[self.config.kpi_col],
            label="Actual KPI",
            linewidth=self.config.plot_linewidth,
            color="green",
            marker="o"
        )
        
        # Plot counterfactual forecast
        plt.plot(
            self.impact_results[self.config.test_period_col],
            self.impact_results["yhat_cf"],
            label="Counterfactual (No Campaign)",
            linewidth=self.config.plot_linewidth,
            linestyle="--",
            color="red",
            marker="s"
        )
        
        # Add confidence interval if enabled
        if self.config.include_confidence_interval:
            plt.fill_between(
                self.impact_results[self.config.test_period_col],
                self.impact_results["yhat_cf_lower"],
                self.impact_results["yhat_cf_upper"],
                alpha=self.config.ci_alpha,
                color="red",
                label="Counterfactual 95% CI"
            )
        
        # Shade the incremental impact area
        plt.fill_between(
            self.impact_results[self.config.test_period_col],
            self.impact_results["yhat_cf"],
            self.impact_results[self.config.kpi_col],
            where=(self.impact_results["incremental_impact"] >= 0),
            alpha=0.2,
            color="green",
            label="Positive Impact"
        )
        
        plt.fill_between(
            self.impact_results[self.config.test_period_col],
            self.impact_results["yhat_cf"],
            self.impact_results[self.config.kpi_col],
            where=(self.impact_results["incremental_impact"] < 0),
            alpha=0.2,
            color="red",
            label="Negative Impact"
        )
        
        # Formatting
        title = title or f"Counterfactual Impact Analysis: {', '.join(self.valid_campaign_regressors)}"
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(self.config.test_period_col, fontsize=12)
        plt.ylabel(self.config.kpi_col, fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Plot saved to {save_path}")
        
        if save_png or save_path:
            png_path = self._resolve_png_path(save_path, "counterfactual_impact.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            self._log(f"Plot saved to {png_path}")

        plt.show()
        
        
    
    def plot_incremental_impact(self,
                               save_path: Optional[str] = None,
                               title: Optional[str] = None,
                               save_png: bool = False) -> None:
        """
        Plot incremental impact over time.
        
        Args:
            save_path (str): Optional path to save the plot (forced to .png)
            title (str): Optional custom title
            save_png (bool): If True, save plot as .png using default name when save_path is None
        """
        if self.impact_results is None:
            print("[ERROR] Analysis not run yet. Call analyze() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Incremental impact
        colors = ['green' if x > 0 else 'red' for x in self.impact_results["incremental_impact"]]
        ax1.bar(
            self.impact_results[self.config.test_period_col],
            self.impact_results["incremental_impact"],
            color=colors,
            alpha=0.7
        )
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_title("Incremental Impact by Period", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Incremental Impact", fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Incremental impact percentage
        valid_pct = self.impact_results[self.impact_results["incremental_impact_pct"].notna()]
        colors_pct = ['green' if x > 0 else 'red' for x in valid_pct["incremental_impact_pct"]]
        ax2.bar(
            valid_pct[self.config.test_period_col],
            valid_pct["incremental_impact_pct"],
            color=colors_pct,
            alpha=0.7
        )
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_title("Incremental Impact %", fontsize=12, fontweight="bold")
        ax2.set_xlabel(self.config.test_period_col, fontsize=11)
        ax2.set_ylabel("Incremental Impact %", fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Plot saved to {save_path}")
        
        if save_png or save_path:
            png_path = self._resolve_png_path(save_path, "incremental_impact.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            self._log(f"Plot saved to {png_path}")

        plt.show()
    
    def export_to_csv(self, filepath: str) -> None:
        """Export impact results to CSV file"""
        if self.impact_results is None:
            print("[ERROR] Analysis not run yet. Call analyze() first.")
            return
        
        self.impact_results.to_csv(filepath, index=False)
        self._log(f"Results exported to {filepath}")
    
    def export_summary_to_json(self, filepath: str) -> None:
        """Export summary to JSON file"""
        if self.impact_summary is None:
            print("[ERROR] Analysis not run yet. Call analyze() first.")
            return
        
        import json
        
        # Convert datetime objects to strings for JSON serialization
        summary_serializable = {}
        for key, value in self.impact_summary.items():
            if hasattr(value, 'isoformat'):
                summary_serializable[key] = value.isoformat()
            elif isinstance(value, list):
                summary_serializable[key] = value
            else:
                summary_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(summary_serializable, f, indent=2)
        
        self._log(f"Summary exported to {filepath}")


# ===== USAGE EXAMPLES =====

if __name__ == "__main__":
    
    print("="*70)
    print("COUNTERFACTUAL IMPACT ANALYSIS - CLASS USAGE EXAMPLES")
    print("="*70)
    
    # Example 1: Basic Usage with Prophet Results
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Counterfactual Analysis")
    print("="*70)
    
    config = {
        'campaign_regressors': ['search_off_weekly', 'social_display_others'],
        'test_period_col': 'ds',
        'kpi_col': 'y',
        'counterfactual_value': 0.0,
        'include_confidence_interval': True,
        'plot_enabled': True,
        'verbose': True
    }
    
    # Initialize analyzer
    analyzer = CounterfactualImpactAnalyzer(config)
    
    # Run analysis (assuming results from prophet_model_run)
    # results = prophet_model_run(df, prophet_config)
    # impact_results = analyzer.analyze(
    #     model=results['model'],
    #     test_data=results['test_data'],
    #     model_regressors=results['config']['regressors_used']
    # )
    
    # Print summary
    # analyzer.print_summary()
    
    # Plot results
    # analyzer.plot_impact()
    # analyzer.plot_incremental_impact()
    
    # Export results
    # analyzer.export_to_csv("impact_results.csv")
    # analyzer.export_summary_to_json("impact_summary.json")
    
    
    # Example 2: Custom Configuration
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Configuration")
    print("="*70)
    
    custom_config = {
        'campaign_regressors': ['tv_spend', 'digital_spend', 'outdoor_spend'],
        'test_period_col': 'date',
        'kpi_col': 'sales',
        'counterfactual_value': 0.0,
        'include_confidence_interval': True,
        'ci_alpha': 0.20,
        'aggregate_level': 'monthly',
        'plot_enabled': True,
        'plot_figsize': (14, 6),
        'plot_linewidth': 2.5,
        'verbose': True
    }
    
    # analyzer = CounterfactualImpactAnalyzer(custom_config)
    
    
    # Example 3: Access Detailed Results
    print("\n" + "="*70)
    print("EXAMPLE 3: Accessing Detailed Results")
    print("="*70)
    
    example_code = """
    # After running analysis
    results = analyzer.get_results()
    
    # Access different components
    impact_df = results['impact_df']
    summary = results['summary']
    config = results['config']
    forecast = results['counterfactual_forecast']
    
    # Get specific metrics
    total_impact = summary['total_incremental_impact']
    lift_pct = summary['incremental_lift_pct']
    
    # View impact table
    print(impact_df[['ds', 'y', 'yhat_cf', 'incremental_impact', 'incremental_impact_pct']])
    
    # Filter periods with significant impact
    significant = impact_df[impact_df['incremental_impact'] > impact_df['incremental_impact'].std()]
    print(f"Periods with significant impact: {len(significant)}")
    """
    
    print(example_code)
    
    
    # Example 4: Multiple Campaign Scenarios
    print("\n" + "="*70)
    print("EXAMPLE 4: Multiple Campaign Scenarios")
    print("="*70)
    
    example_multi = """
    # Scenario 1: All campaigns
    config_all = {
        'campaign_regressors': ['search', 'social', 'display', 'tv'],
        'verbose': True
    }
    analyzer_all = CounterfactualImpactAnalyzer(config_all)
    results_all = analyzer_all.analyze(model, test_data, model_regressors)
    
    # Scenario 2: Only digital campaigns
    config_digital = {
        'campaign_regressors': ['search', 'social', 'display'],
        'verbose': True
    }
    analyzer_digital = CounterfactualImpactAnalyzer(config_digital)
    results_digital = analyzer_digital.analyze(model, test_data, model_regressors)
    
    # Scenario 3: Only offline campaigns
    config_offline = {
        'campaign_regressors': ['tv', 'outdoor'],
        'verbose': True
    }
    analyzer_offline = CounterfactualImpactAnalyzer(config_offline)
    results_offline = analyzer_offline.analyze(model, test_data, model_regressors)
    
    # Compare scenarios
    print(f"All campaigns impact: {results_all['summary']['total_incremental_impact']}")
    print(f"Digital campaigns impact: {results_digital['summary']['total_incremental_impact']}")
    print(f"Offline campaigns impact: {results_offline['summary']['total_incremental_impact']}")
    """
    
    print(example_multi)