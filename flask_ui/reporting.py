from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape
import base64

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _data_url_image(src: str):
    if not isinstance(src, str) or not src.startswith("data:image/") or "," not in src:
        return None
    try:
        raw = base64.b64decode(src.split(",", 1)[1], validate=True)
    except Exception:
        return None
    image = Image(BytesIO(raw))
    max_width = 155 * mm
    max_height = 105 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def _metric_rows(result: dict) -> list[list[str]]:
    metrics = result.get("metrics") or {}
    calibration = result.get("calibration") or {}

    blemish = metrics.get("blemish_percentage")
    try:
        blemish_text = f"{float(blemish):.2f}%"
    except (TypeError, ValueError):
        blemish_text = "Unavailable"

    if calibration.get("applied"):
        try:
            size_text = (
                f"{float(calibration['length_cm']):.1f} × "
                f"{float(calibration['width_cm']):.1f} cm"
            )
        except (KeyError, TypeError, ValueError):
            size_text = "Unavailable"
    else:
        size_text = "N/A"

    return [
        ["Ripeness", str(metrics.get("ripeness") or "Unavailable")],
        ["Visible blemish", blemish_text],
        ["Surface quality", str(metrics.get("surface_quality") or "Unavailable")],
        ["Estimated size", size_text],
    ]


def build_report(results):
    successful = [r for r in results if isinstance(r, dict) and not r.get("error")]
    if not successful:
        raise ValueError("No successful assessment results are available for PDF export.")

    output = BytesIO()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17372c"),
        spaceAfter=10,
    )
    name_style = ParagraphStyle(
        "ImageName",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#215d43"),
        spaceAfter=10,
    )

    story = [Paragraph("Banana Quality Assessment", title_style)]
    for index, result in enumerate(successful):
        if index:
            story.append(PageBreak())

        story.append(Paragraph(escape(str(result.get("filename") or "Image")), name_style))

        image = _data_url_image((result.get("images") or {}).get("overlay", ""))
        if image is not None:
            image.hAlign = "CENTER"
            story.extend([image, Spacer(1, 10)])

        rows = [["Measure", "Finding"], *_metric_rows(result)]
        table = Table(
            [[Paragraph(escape(str(value)), styles["Normal"]) for value in row] for row in rows],
            colWidths=[55 * mm, 95 * mm],
            hAlign="CENTER",
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf4ed")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17372c")),
            ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#cfdcd4")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(table)

    SimpleDocTemplate(
        output,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    ).build(story)
    return output.getvalue()
