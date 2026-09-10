#!/usr/bin/env python3
"""Place two one-page PDFs side by side without rasterizing them."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject


MM_TO_PT = 72.0 / 25.4


def one_page(path: Path) -> PageObject:
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected one-page PDF: {path}")
    return reader.pages[0]


def write_horizontal(
    sources: list[Path],
    output: Path,
    panel_width_mm: float,
    gap_mm: float,
) -> None:
    pages = [one_page(path) for path in sources]
    panel_width_pt = panel_width_mm * MM_TO_PT
    gap_pt = gap_mm * MM_TO_PT
    heights = [
        float(page.mediabox.height)
        * panel_width_pt
        / float(page.mediabox.width)
        for page in pages
    ]
    width_pt = panel_width_pt * len(pages) + gap_pt * (len(pages) - 1)
    height_pt = max(heights)
    canvas = PageObject.create_blank_page(width=width_pt, height=height_pt)
    x_pt = 0.0
    for page, height in zip(pages, heights):
        scale = panel_width_pt / float(page.mediabox.width)
        canvas.merge_transformed_page(
            page,
            Transformation().scale(scale).translate(
                x_pt,
                (height_pt - height) / 2.0,
            ),
            expand=False,
        )
        x_pt += panel_width_pt + gap_pt
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(canvas)
    with output.open("wb") as handle:
        writer.write(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel_a", type=Path)
    parser.add_argument("panel_b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--panel-width-mm", type=float, default=89.0)
    parser.add_argument("--gap-mm", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_horizontal(
        [args.panel_a, args.panel_b],
        args.output,
        panel_width_mm=args.panel_width_mm,
        gap_mm=args.gap_mm,
    )


if __name__ == "__main__":
    main()
