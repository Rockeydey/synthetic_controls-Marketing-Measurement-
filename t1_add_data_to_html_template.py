"""
Updated : 2026-06-29

"""


from src.to_base64 import apply_base64_charts_to_html_file


html_path = r"dashboard\dev_template.html"

chart_map = {
    "overviewChart": r"outputs\decomposition_stacked.png",
    "seasonalityPatternChart": r"outputs\seasonality_pattern.png",
    "actualPredictedChart": r"outputs\actual_vs_predicted.png",
    "counterfactualChart": r"outputs\counterfactual_impact.png",
    "forecastChart": r"outputs\forecast_chart.png",
    "counterfactualIncrementalChart": r"outputs\counterfactual_incremental.png",
    # "componentsChart": r"c:\path\to\components.png",
    # "cvChart": r"c:\path\to\cv.png",
    # "residualHistogram": r"c:\path\to\residual_hist.png",
    # "residualScatter": r"c:\path\to\residual_scatter.png",
    # "campaignBreakdown": r"c:\path\to\campaign_breakdown.png",
}

# apply_base64_charts_to_html_file(
#         html_file_path=html_path,
#         chart_image_map=chart_map,
#         output_file_path=None,          # overwrite same file
#         keep_hidden_canvas=True,        # safe for existing JS
#         create_backup=False,                 # no backup
#     )

# apply_base64_charts_to_html_file(
#             html_file_path=html_path,
#             chart_image_map=chart_map,
#             output_file_path=None,      # overwrite same file
#             keep_hidden_canvas=True,    # safe for existing JS
#             create_backup=False,
#             default_width="900px",      # default width for charts without specific size mapping
#             default_height="420px",     # default height for charts without specific size mapping
            
#     )

apply_base64_charts_to_html_file(
            html_file_path=html_path,
            chart_image_map=chart_map,
            keep_hidden_canvas=True,
            create_backup=False,
            default_width="900px",
            default_height="420px",
            default_align="center",
            chart_align_map={
                "overviewChart": "center",
                "seasonalityPatternChart": "center",
                "counterfactualChart": "center",
                "counterfactualIncrementalChart": "center",
                "actualPredictedChart": "center",
            },
            chart_size_map={
                "overviewChart": {"width": "900px", "height": "420px"},
                "seasonalityPatternChart": {"width": "800px", "height": "360px"},
                "counterfactualChart": {"width": "800px", "height": "360px"},
                "counterfactualIncrementalChart": {"width": "900px", "height": "420px"},
                "actualPredictedChart": {"width": "800px", "height": "360px"},
                "forecastChart": {"width": "800px", "height": "360px"},
                
            },
        )