"""
Updated: 2026-07-01

"""


import base64
from pathlib import Path
import re
from typing import Dict, Union, Optional

def _image_to_data_uri(image_path: Union[str, Path]) -> str:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    ext = image_path.suffix.lower()
    if ext not in mime_types:
        raise ValueError(f"Unsupported image format: {ext}")

    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_types[ext]};base64,{b64}"




# ...existing code...
import base64
from pathlib import Path
import re
from typing import Dict, Union, Optional
# ...existing code...

def replace_canvas_with_base64_images(
    html_text: str,
    chart_image_map: Dict[str, Union[str, Path]],
    keep_hidden_canvas: bool = True,
    default_width: str = "100%",
    default_height: str = "100%",
    chart_size_map: Optional[Dict[str, Dict[str, str]]] = None,
    default_align: str = "center",
    chart_align_map: Optional[Dict[str, str]] = None,
) -> str:
    updated_html = html_text
    chart_size_map = chart_size_map or {}
    chart_align_map = chart_align_map or {}

    for chart_id, image_path in chart_image_map.items():
        data_uri = _image_to_data_uri(image_path)

        size = chart_size_map.get(chart_id, {})
        width = size.get("width", default_width)
        height = size.get("height", default_height)

        align = chart_align_map.get(chart_id, default_align).strip().lower()
        if align not in {"left", "center", "right"}:
            align = "center"

        if align == "left":
            margin_style = "margin:0 auto 0 0;"
        elif align == "right":
            margin_style = "margin:0 0 0 auto;"
        else:
            margin_style = "margin:0 auto;"

        # Keep chart inside tab/container width
        wrapper_style = (
            f"width:min(100%,{width});"
            f"height:{height};"
            f"max-width:100%;"
            f"{margin_style}"
        )

        img_tag = (
            f'<img src="{data_uri}" alt="{chart_id}" '
            f'style="width:100%;height:100%;max-width:100%;object-fit:contain;display:block;" />'
        )

        if keep_hidden_canvas:
            replacement = (
                f'<div class="embedded-chart" data-chart-id="{chart_id}" style="{wrapper_style}">'
                f"{img_tag}"
                f'<canvas id="{chart_id}" data-embedded-base64="true" style="display:none !important;"></canvas>'
                f"</div>"
            )
        else:
            replacement = (
                f'<div class="embedded-chart" data-chart-id="{chart_id}" style="{wrapper_style}">'
                f"{img_tag}"
                f"</div>"
            )

        # 1) If already embedded, replace whole wrapper (idempotent update)
        embedded_pattern = re.compile(
            rf'<div\b[^>]*\bclass=["\'][^"\']*\bembedded-chart\b[^"\']*["\'][^>]*'
            rf'\bdata-chart-id=["\']{re.escape(chart_id)}["\'][^>]*>.*?</div>',
            flags=re.IGNORECASE | re.DOTALL
        )
        updated_html, count = embedded_pattern.subn(replacement, updated_html, count=1)

        # 2) Else replace original canvas (but skip marked embedded canvas)
        if count == 0:
            canvas_pattern = re.compile(
                rf'<canvas\b'
                rf'(?=[^>]*\bid=["\']{re.escape(chart_id)}["\'])'
                rf'(?![^>]*\bdata-embedded-base64=["\']true["\'])'
                rf'[^>]*>\s*</canvas>',
                flags=re.IGNORECASE
            )
            updated_html, count = canvas_pattern.subn(replacement, updated_html, count=1)

        if count == 0:
            print(f"[WARN] canvas/wrapper for id='{chart_id}' not found in HTML.")

    return updated_html


def apply_base64_charts_to_html_file(
    html_file_path: Union[str, Path],
    chart_image_map: Dict[str, Union[str, Path]],
    output_file_path: Union[str, Path, None] = None,
    keep_hidden_canvas: bool = True,
    create_backup: bool = True,
    default_width: str = "100%",
    default_height: str = "100%",
    chart_size_map: Optional[Dict[str, Dict[str, str]]] = None,
    default_align: str = "center",
    chart_align_map: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Apply replacements to an HTML file.
    """
    html_file_path = Path(html_file_path)
    if not html_file_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")

    original = html_file_path.read_text(encoding="utf-8")

    updated = replace_canvas_with_base64_images(
        html_text=original,
        chart_image_map=chart_image_map,
        keep_hidden_canvas=keep_hidden_canvas,
        default_width=default_width,
        default_height=default_height,
        chart_size_map=chart_size_map,
        default_align=default_align,
        chart_align_map=chart_align_map,
    )

    if output_file_path is None:
        output_file_path = html_file_path
    output_file_path = Path(output_file_path)

    if create_backup and output_file_path == html_file_path:
        backup_path = html_file_path.with_suffix(html_file_path.suffix + ".bak")
        backup_path.write_text(original, encoding="utf-8")
        print(f"[INFO] Backup created: {backup_path}")

    output_file_path.write_text(updated, encoding="utf-8")
    print(f"[INFO] Updated HTML written to: {output_file_path}")
    return output_file_path


if __name__ == "__main__":
    # Example: apply to one or many charts in dev_template.html
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
            keep_hidden_canvas=True,
            create_backup=False,
            default_width="900px",
            default_height="420px",
            default_align="center",
            chart_align_map={
                "overviewChart": "center",
                "counterfactualChart": "left",
                "actualPredictedChart": "right",
            },
            chart_size_map={
                "overviewChart": {"width": "1000px", "height": "500px"},
                "actualPredictedChart": {"width": "800px", "height": "360px"},
            },
        )