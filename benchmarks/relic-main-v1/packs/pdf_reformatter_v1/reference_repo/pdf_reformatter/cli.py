"""Command-line entry point for the reformatter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from pdf_reformatter.extract import Row, extract_rows
from pdf_reformatter.layout import (
    DEFAULT_BOTTOM_MARGIN,
    DEFAULT_FONT_SIZE,
    DEFAULT_LEADING,
    DEFAULT_PAGE_HEIGHT,
    DEFAULT_PAGE_WIDTH,
    DEFAULT_TOP_MARGIN,
    paginate,
)
from pdf_reformatter.writers import write_csv, write_docx, write_pdf


def _parse_widths(spec: str | None, n_columns: int, page_width: float) -> List[float]:
    if spec:
        widths = [float(x) for x in spec.split(",") if x.strip()]
        if len(widths) != n_columns:
            raise SystemExit(
                f"--col-widths has {len(widths)} entries but the PDF has "
                f"{n_columns} columns"
            )
        return widths
    # Equal-share fallback inside a centred body area.
    body = page_width - 2 * DEFAULT_TOP_MARGIN
    if body <= 0 or n_columns == 0:
        return [page_width / max(n_columns, 1)] * max(n_columns, 1)
    return [body / n_columns] * n_columns


def reformat(
    input_pdf: str | Path,
    output_pdf: str | Path,
    output_docx: str | Path,
    output_csv: str | Path,
    col_widths: Sequence[float] | None = None,
    page_width: float = DEFAULT_PAGE_WIDTH,
    page_height: float = DEFAULT_PAGE_HEIGHT,
    top_margin: float = DEFAULT_TOP_MARGIN,
    bottom_margin: float = DEFAULT_BOTTOM_MARGIN,
    font_size: float = DEFAULT_FONT_SIZE,
    leading: float = DEFAULT_LEADING,
    repeat_header: bool = False,
) -> dict:
    """Library entry point. Returns a dict of the four output paths."""

    rows: List[Row] = extract_rows(str(input_pdf))
    if not rows:
        # Still emit empty deliverables so downstream callers see all
        # three files exist.
        Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
        Path(output_pdf).write_bytes(b"%PDF-1.4\n%%EOF\n")
        write_docx([], output_docx)
        write_csv([], output_csv)
        return {
            "pdf": Path(output_pdf),
            "docx": Path(output_docx),
            "csv": Path(output_csv),
            "rows": 0,
            "pages": 0,
        }
    n_cols = len(rows[0].cells)
    widths = list(col_widths) if col_widths is not None else _parse_widths(
        None, n_cols, page_width
    )
    pages = paginate(
        rows,
        col_widths=widths,
        page_height=page_height,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        font_size=font_size,
        leading=leading,
        repeat_header=repeat_header,
    )
    write_pdf(
        pages,
        output_pdf,
        col_widths=widths,
        page_width=page_width,
        page_height=page_height,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        font_size=font_size,
        leading=leading,
    )
    write_docx(rows, output_docx)
    write_csv(rows, output_csv)
    return {
        "pdf": Path(output_pdf),
        "docx": Path(output_docx),
        "csv": Path(output_csv),
        "rows": len(rows),
        "pages": len(pages),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf-reformatter",
        description=(
            "Re-flow an overflowing tabular PDF so no row straddles a "
            "page boundary, and emit PDF, DOCX, and CSV outputs."
        ),
    )
    p.add_argument("input", help="Path to the input PDF.")
    p.add_argument(
        "--out-pdf", required=True, help="Where to write the reformatted PDF."
    )
    p.add_argument("--out-docx", required=True, help="Where to write the DOCX output.")
    p.add_argument("--out-csv", required=True, help="Where to write the CSV output.")
    p.add_argument(
        "--col-widths",
        default=None,
        help="Comma-separated column widths in PDF points. Defaults to equal-share.",
    )
    p.add_argument("--page-width", type=float, default=DEFAULT_PAGE_WIDTH)
    p.add_argument("--page-height", type=float, default=DEFAULT_PAGE_HEIGHT)
    p.add_argument("--top-margin", type=float, default=DEFAULT_TOP_MARGIN)
    p.add_argument("--bottom-margin", type=float, default=DEFAULT_BOTTOM_MARGIN)
    p.add_argument("--font-size", type=float, default=DEFAULT_FONT_SIZE)
    p.add_argument("--leading", type=float, default=DEFAULT_LEADING)
    p.add_argument("--repeat-header", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = extract_rows(args.input)
    n_cols = len(rows[0].cells) if rows else 0
    widths = _parse_widths(args.col_widths, n_cols, args.page_width) if n_cols else []
    result = reformat(
        args.input,
        args.out_pdf,
        args.out_docx,
        args.out_csv,
        col_widths=widths or None,
        page_width=args.page_width,
        page_height=args.page_height,
        top_margin=args.top_margin,
        bottom_margin=args.bottom_margin,
        font_size=args.font_size,
        leading=args.leading,
        repeat_header=args.repeat_header,
    )
    print(
        f"Wrote {result['rows']} rows across {result['pages']} pages: "
        f"{result['pdf']}, {result['docx']}, {result['csv']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
