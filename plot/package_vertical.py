"""Place one-page PDFs vertically while preserving vectors."""
from pathlib import Path
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
MM_TO_PT = 72.0 / 25.4
def source_page(path: Path) -> PageObject:
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected one-page PDF: {path}")
    return reader.pages[0]

def place_page(
    canvas: PageObject,
    page: PageObject,
    target_width_pt: float,
    x_pt: float,
    y_pt: float,
) -> float:
    source_width = float(page.mediabox.width)
    source_height = float(page.mediabox.height)
    scale = target_width_pt / source_width
    canvas.merge_transformed_page(
        page,
        Transformation().scale(scale).translate(x_pt, y_pt),
        expand=False,
    )
    return source_height * scale

def write_vertical(
    sources: list[Path],
    output: Path,
    panel_width_mm: float,
    gap_mm: float,
) -> dict[str, object]:
    pages = [source_page(path) for path in sources]
    panel_width_pt = panel_width_mm * MM_TO_PT
    gap_pt = gap_mm * MM_TO_PT
    heights = [
        float(page.mediabox.height)
        * panel_width_pt
        / float(page.mediabox.width)
        for page in pages
    ]
    width_pt = panel_width_pt
    height_pt = sum(heights) + gap_pt * (len(pages) - 1)
    canvas = PageObject.create_blank_page(width=width_pt, height=height_pt)
    y_top = height_pt
    for page, height in zip(pages, heights):
        y_top -= height
        place_page(canvas, page, panel_width_pt, 0.0, y_top)
        y_top -= gap_pt
    writer = PdfWriter()
    writer.add_page(canvas)
    with output.open("wb") as handle:
        writer.write(handle)
    return {
        "output": str(output),
        "layout": "vertical",
        "sources": [str(path) for path in sources],
        "panel_width_mm": panel_width_mm,
        "gap_mm": gap_mm,
        "page_width_mm": width_pt / MM_TO_PT,
        "page_height_mm": height_pt / MM_TO_PT,
    }
