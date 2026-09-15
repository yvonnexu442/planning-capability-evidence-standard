"""Utilities for exporting experiment outputs into LaTeX-ready assets."""

from pathlib import Path
from shutil import copy2
from typing import Dict, Iterable, List, Mapping, Optional, Union

import pandas as pd


PathLike = Union[str, Path]


def prepare_dataframe_for_latex(
    data: pd.DataFrame,
    numeric_precision: int = 3,
    column_renames: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Return a display-ready DataFrame for LaTeX export.

    The function preserves English column names by default and optionally
    applies paper-friendly English display names. Numeric columns are rounded
    consistently so tables can be compared across experiments.

    This utility is part of the paper workflow: Python experiments produce raw
    result tables, and this function prepares them for direct LaTeX inclusion.
    """
    prepared = data.copy()
    numeric_columns = prepared.select_dtypes(include=["number"]).columns
    prepared.loc[:, numeric_columns] = prepared.loc[:, numeric_columns].round(int(numeric_precision))
    if column_renames is not None:
        prepared = prepared.rename(columns=dict(column_renames))
    return prepared


def dataframe_to_latex_table(
    data: pd.DataFrame,
    caption: str,
    label: str,
    index: bool = False,
    numeric_precision: int = 3,
    column_renames: Optional[Mapping[str, str]] = None,
    booktabs: bool = True,
    resize_to_textwidth: bool = False,
) -> str:
    """Return a LaTeX table string using booktabs-style rules.

    The table is suitable for direct inclusion in a manuscript. Pandas writes
    LaTeX tables with `\\toprule`, `\\midrule`, and `\\bottomrule`, which are
    provided by the LaTeX `booktabs` package.

    The `booktabs` argument is kept explicit because the repository standard is
    to export manuscript tables in booktabs format.
    """
    if not booktabs:
        raise ValueError("This repository exports LaTeX tables with booktabs=True.")

    prepared = prepare_dataframe_for_latex(
        data,
        numeric_precision=numeric_precision,
        column_renames=column_renames,
    )
    float_format = "{:." + str(int(numeric_precision)) + "f}"
    try:
        latex_table = prepared.to_latex(
            index=index,
            caption=caption,
            label=label,
            escape=True,
            multicolumn=True,
            multirow=True,
            float_format=lambda value: float_format.format(value),
        )
    except ImportError as error:
        if "jinja2" not in str(error).lower():
            raise
        latex_table = _dataframe_to_latex_without_jinja(
            prepared, caption=caption, label=label, index=index, float_format=float_format
        )
    if resize_to_textwidth:
        latex_table = _wrap_tabular_in_resizebox(latex_table)
    return latex_table


def _dataframe_to_latex_without_jinja(data, caption, label, index, float_format):
    """Provide a deterministic booktabs fallback for minimal runtimes."""
    def escape(value):
        text = str(value)
        for source, target in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
            text = text.replace(source, target)
        return text

    columns = list(data.columns)
    headers = ([escape(data.index.name or "")] if index else []) + [escape(value) for value in columns]
    rows = []
    for row_index, row in data.iterrows():
        values = [row_index] + row.tolist() if index else row.tolist()
        rendered = [float_format.format(value) if isinstance(value, float) else escape(value) for value in values]
        rows.append(" & ".join(rendered) + r" \\")
    alignment = "l" * len(headers)
    return "\n".join([
        r"\begin{table}",
        r"\caption{" + escape(caption) + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def export_dataframe_to_latex(
    data: pd.DataFrame,
    tex_path: PathLike,
    caption: str,
    label: str,
    csv_path: Optional[PathLike] = None,
    index: bool = False,
    numeric_precision: int = 3,
    column_renames: Optional[Mapping[str, str]] = None,
    booktabs: bool = True,
    resize_to_textwidth: bool = False,
) -> Dict[str, Path]:
    """Save a DataFrame as both CSV and LaTeX table files.

    Major result tables should be available as `.csv` for auditability and as
    `.tex` for direct manuscript input. This function writes both versions and
    returns their paths.

    Captions and labels should be English because all repository artifacts are
    intended for an English-language research paper.
    """
    tex_output = Path(tex_path)
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = Path(csv_path) if csv_path is not None else tex_output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepare_dataframe_for_latex(
        data,
        numeric_precision=numeric_precision,
        column_renames=column_renames,
    )
    prepared.to_csv(csv_output, index=index)
    latex_table = dataframe_to_latex_table(
        data,
        caption=caption,
        label=label,
        index=index,
        numeric_precision=numeric_precision,
        column_renames=column_renames,
        booktabs=booktabs,
        resize_to_textwidth=resize_to_textwidth,
    )
    tex_output.write_text(latex_table, encoding="utf-8")
    return {"csv": csv_output, "tex": tex_output}


def export_summary_table(
    data: pd.DataFrame,
    table_name: str,
    output_dir: PathLike = "paper/tables",
    caption: Optional[str] = None,
    label: Optional[str] = None,
    numeric_precision: int = 3,
    column_renames: Optional[Mapping[str, str]] = None,
    resize_to_textwidth: bool = False,
) -> Dict[str, Path]:
    """Export a named summary table into the paper table directory.

    This wrapper standardizes paper table file naming. For example,
    `forecast_metrics_table` creates `forecast_metrics_table.csv` and
    `forecast_metrics_table.tex` under `paper/tables/`.
    """
    table_stem = Path(table_name).stem
    output_path = Path(output_dir)
    caption_text = caption or _title_from_stem(table_stem)
    label_text = label or "tab:{}".format(table_stem.replace("_", "-"))
    return export_dataframe_to_latex(
        data=data,
        tex_path=output_path / "{}.tex".format(table_stem),
        csv_path=output_path / "{}.csv".format(table_stem),
        caption=caption_text,
        label=label_text,
        numeric_precision=numeric_precision,
        column_renames=column_renames,
        resize_to_textwidth=resize_to_textwidth,
    )


def copy_figure_file(
    source_path: PathLike,
    target_dir: PathLike = "paper/figures",
    output_name: Optional[str] = None,
) -> Path:
    """Copy a figure asset into the paper figure directory.

    Research figures should be saved as PDF for LaTeX compatibility. PNG files
    may also be copied for quick viewing. The function does not fabricate or
    convert figures; it only moves available experiment assets into the paper
    asset directory.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError("Figure file does not exist: {}".format(source))
    allowed_suffixes = {".pdf", ".png", ".jpg", ".jpeg"}
    if source.suffix.lower() not in allowed_suffixes:
        raise ValueError("Unsupported figure extension: {}".format(source.suffix))

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    destination = target / (output_name or source.name)
    copy2(str(source), str(destination))
    return destination


def latex_figure_snippet(
    figure_path: PathLike,
    caption: str,
    label: str,
    width: str = "0.9\\linewidth",
) -> str:
    """Return a LaTeX figure snippet for an exported figure.

    The snippet can be pasted into a section file or saved by an experiment
    script. It assumes the figure path is relative to the LaTeX paper directory.
    """
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\includegraphics[width="
        + width
        + "]{"
        + str(figure_path)
        + "}\n"
        "\\caption{"
        + caption
        + "}\n"
        "\\label{"
        + label
        + "}\n"
        "\\end{figure}\n"
    )


def latex_table_snippet(table_path: PathLike) -> str:
    """Return a LaTeX input snippet for an exported table file.

    Exported table files already contain captions and labels, so the manuscript
    can include them with a single `\\input{...}` command.
    """
    return "\\input{" + str(table_path) + "}\n"


def write_asset_manifest(
    manifest_path: PathLike,
    table_paths: Iterable[PathLike],
    figure_paths: Iterable[PathLike],
) -> Path:
    """Write a Markdown manifest of LaTeX-ready paper assets.

    The manifest helps paper authors see which generated tables and figures are
    available for direct inclusion in the manuscript.
    """
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table_list = [Path(path) for path in table_paths]
    figure_list = [Path(path) for path in figure_paths]

    lines: List[str] = ["# Paper Asset Manifest", ""]
    lines.append("## Tables")
    if table_list:
        for path in sorted(table_list):
            lines.append("- `{}`".format(path.as_posix()))
    else:
        lines.append("- No LaTeX-ready tables are available yet.")
    lines.append("")
    lines.append("## Figures")
    if figure_list:
        for path in sorted(figure_list):
            lines.append("- `{}`".format(path.as_posix()))
    else:
        lines.append("- No paper figures are available yet.")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _title_from_stem(stem: str) -> str:
    """Convert a snake_case file stem into a readable English title."""
    return stem.replace("_", " ").title()


def _wrap_tabular_in_resizebox(latex_table: str) -> str:
    """Wrap the tabular block in a text-width resizebox."""
    begin_marker = "\\begin{tabular}"
    end_marker = "\\end{tabular}"
    begin_position = latex_table.find(begin_marker)
    end_position = latex_table.find(end_marker)
    if begin_position == -1 or end_position == -1:
        return latex_table
    end_position = end_position + len(end_marker)
    return (
        latex_table[:begin_position]
        + "\\resizebox{\\textwidth}{!}{%\n"
        + latex_table[begin_position:end_position]
        + "\n}\n"
        + latex_table[end_position:]
    )
