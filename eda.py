import os

import pandas as pd
from owlmix.reporting import ReportBuilder, ReportHTMLRenderer

csv_file = "data\\combined_data.csv"

df = pd.read_csv(csv_file)
report_builder = ReportBuilder(
    df=df, 
    target_col="sales", 
    date_col="month"
)

# Update the config, if needed
report_builder.config.update_config(
    acf_pacf={
        "columns": ["sales"],
        "n_lags": 5
    },
    vif={
        "features": ['dig_aud','video_io']
    },
    # correlation={
    #     "columns": ["sales", "tv_spend"],
    #     "n_lags": 8,
    #     "precision": 5
    # },
    box_plot={
        "columns": ['social_video'],
        "n_plot_per_row": 3,
        "method": "zscore",
        "threshold": 1.5  # default is 3 for method "zscore" and 1.5 for method "iqr"
    },
    # ccf={
    #     "feature_columns": ['tv_spend'],
    #     "max_lag": 3
    # },
    # update the other configs
)


# report_builder.exclude_sections(["vif", "ccf"], verbose=True)
# report_builder.include_sections([SectionEnum.CAUSALITY, "ccf"], verbose=True)

report_builder.add_section_by_name("acf_pacf", verbose=True)
report_builder.add_section_by_name("correlation", verbose=True)
report_builder.add_section_by_name("ccf", verbose=True)
report_builder.add_section_by_name("dist_numeric", verbose=True)


# report_builder.add_all_sections(verbose=True)
report = report_builder.build()
report_builder.save("result.json")

# r"outputs\\result.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_file = os.path.join(BASE_DIR, "outputs", "result.json")

renderer = ReportHTMLRenderer()
html_str = renderer.render_from_json(json_file)
renderer.save_html(html_str,os.path.join(BASE_DIR, "outputs", "report.html"))