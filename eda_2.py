import warnings
import sys
from pathlib import Path
 
sys.dont_write_bytecode = True
warnings.simplefilter('ignore', category=UserWarning)
 
import pandas as pd
from owlmix.utils.sample_data_generator import create_sample_data
from owlmix.reporting import ReportBuilder, ReportHTMLRenderer
from owlmix.typing.enums import SectionEnum
 
csv_file = r"data\\combined_data.csv"

def render_html_report_from_json():
    report_json_path = "outputs/report.json"
    saved_html_path = "outputs/report.html"
    
    renderer = ReportHTMLRenderer()
    html_str = renderer.render_from_json(report_json_path)
    renderer.save_html(html_str, saved_html_path)
 
 
def load_data_from_csv(csv_path: str = csv_file) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(csv_path)
 
 
def main():
    # Load data from CSV
    df = load_data_from_csv()
    columns_to_keep = [
        "month",  # date/time column
        "sales",  # target column
        "display_io",
        "display_prog",
        "social_display",
        "social_video",
        "search_off_weekly",
        "search_def_weekly",
        "video_prog_fm",
        "video_prog_nm",
        "display_io_fm",
        "display_io_nm",
        "display_io_im",
        "display_prog_fm",
        "display_prog_nm",
        "display_prog_im",
        "social_display_fm",
        "social_display_nm",
        "social_display_im",
        "social_video_fm",
        "social_video_nm",
        "video_prog_dv360",
        "display_prog_dv360",
        "social_display_meta",
        "social_video_meta",
        "social_display_others",
        "social_video_others",
        "ooh_fm_spend",
    ]
    df = df[columns_to_keep].copy()
    print(f"Data loaded from CSV: {df.shape[0]} rows, {df.shape[1]} columns")
 
    report_builder = ReportBuilder(
        df=df,
        target_col="sales",
        date_col="month"
    )
 
    report_builder.config.update_config(
        acf_pacf={
            "columns": ["sales", "display_io", "social_video"],
            "n_lags": 5
        },
        box_plot={
            "columns": ["display_io", "display_prog", "social_display", "social_video", "video_prog_fm", "video_prog_nm"],
        },
        vif={
            "features": [
                "social_video_nm",
                "search_off_weekly",
                "search_def_weekly",
                "video_prog_fm",
                "video_prog_nm",
            ]
        },
        response_summary={
            "feature_columns": [
                "display_io",
                "display_prog",
                "social_display",
                "social_video",
                "video_prog_fm",
                "video_prog_nm",
                "display_io_fm",
                "display_io_nm",
                "display_io_im",
                "display_prog_fm",
                "display_prog_nm",
                "display_prog_im",
                "social_display_fm",
                "social_display_nm",
                "social_display_im",
                "social_video_fm",
                "social_video_nm",
                "video_prog_dv360",
                "display_prog_dv360",
                "social_display_meta",
                "social_video_meta",
                "social_display_others",
                "social_video_others",
                "ooh_fm_spend",
            ]
        }
    )
 
    report_builder.add_all_sections(verbose=True)
    # report_builder.exclude_sections([SectionEnum.VIF], verbose=True)
    report_builder.add_output_dir("outputs")
    report = report_builder.build(
        with_all_sections=True,
        verbose=True,
    )
    report_builder.save("report.json")
 
 
if __name__ == "__main__":
    main()
    render_html_report_from_json()