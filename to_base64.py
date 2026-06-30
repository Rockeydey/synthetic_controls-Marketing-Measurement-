import base64
from pathlib import Path
import os
import re
from typing import Dict, Union

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


def replace_canvas_with_base64_images(
    html_text: str,
    chart_image_map: Dict[str, Union[str, Path]],
    keep_hidden_canvas: bool = True
) -> str:
    """
    Replace <canvas id="..."></canvas> with Base64 <img> for each chart id.
    If keep_hidden_canvas=True, keeps the original canvas hidden to avoid JS Chart errors.
    """
    updated_html = html_text

    for chart_id, image_path in chart_image_map.items():
        data_uri = _image_to_data_uri(image_path)
        img_tag = (
            f'<img src="{data_uri}" alt="{chart_id}" '
            f'style="max-width:100%;max-height:100%;object-fit:contain;display:block;margin:auto;" />'
        )

        if keep_hidden_canvas:
            replacement = (
                f'<div class="embedded-chart" style="width:100%;height:100%;">'
                f"{img_tag}"
                f'<canvas id="{chart_id}" style="display:none !important;"></canvas>'
                f"</div>"
            )
        else:
            replacement = img_tag

        # Match canvas by exact id, regardless of extra attributes
        pattern = re.compile(
            rf'<canvas\b(?=[^>]*\bid=["\']{re.escape(chart_id)}["\'])[^>]*>\s*</canvas>',
            flags=re.IGNORECASE
        )

        updated_html, count = pattern.subn(replacement, updated_html, count=1)
        if count == 0:
            print(f"[WARN] canvas id='{chart_id}' not found in HTML.")

    return updated_html


def apply_base64_charts_to_html_file(
    html_file_path: Union[str, Path],
    chart_image_map: Dict[str, Union[str, Path]],
    output_file_path: Union[str, Path, None] = None,
    keep_hidden_canvas: bool = True,
    create_backup: bool = True
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
        keep_hidden_canvas=keep_hidden_canvas
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
        output_file_path=None,          # overwrite same file
        keep_hidden_canvas=True,        # safe for existing JS
        create_backup=True
    )