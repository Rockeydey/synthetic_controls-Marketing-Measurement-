"""
Updated : 2026-06-29

"""


from to_base64 import apply_base64_charts_to_html_file


html_path = r"dashboard\dev_template.html"

chart_map = {
    "overviewChart": r"outputs\prophet.png",
    "actualPredictedChart": r"outputs\actual_vs_predicted.png",
    # "counterfactualChart": r"c:\path\to\counterfactual.png",
    # "forecastChart": r"c:\path\to\forecast.png",
    # "componentsChart": r"c:\path\to\components.png",
    # "cvChart": r"c:\path\to\cv.png",
    # "residualHistogram": r"c:\path\to\residual_hist.png",
    # "residualScatter": r"c:\path\to\residual_scatter.png",
    # "campaignBreakdown": r"c:\path\to\campaign_breakdown.png",
}

apply_base64_charts_to_html_file(
        html_file_path=html_path,
        chart_image_map=chart_map,
        output_file_path=None,          # overwrite same file
        keep_hidden_canvas=True,        # safe for existing JS
        create_backup=True
    )