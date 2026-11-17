from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_DIR = Path("vector_output")

SCRIPTS = [
    "algorithm_comparison.py",
    "batch_size_comparison.py",
    "longcontext_data.py",
    "convergence.py",
    "scaling_data.py",
]

FIGURE_BASENAMES = {
    "batch_size_total_throughput_final",
    "context_length_comparison_with_oom_label",
    "grpo_performance_comparison",
    "ppo_performance_comparison",
    "reward_and_entropy_comparison_final",
    "scaling_trend_new",
}


def _target_svg(path: Path) -> Path | None:
    if path.stem not in FIGURE_BASENAMES:
        return None
    if path.suffix.lower() not in {".pdf", ".png"}:
        return None
    return OUTPUT_DIR / f"{path.stem}.svg"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    original_savefig = plt.savefig
    written_svgs: set[Path] = set()

    def savefig_with_svg(fname, *args, **kwargs):
        result = original_savefig(fname, *args, **kwargs)
        svg_path = _target_svg(Path(fname))
        if svg_path is not None and svg_path not in written_svgs:
            svg_kwargs = dict(kwargs)
            svg_kwargs.pop("format", None)
            original_savefig(svg_path, *args, format="svg", **svg_kwargs)
            written_svgs.add(svg_path)
        return result

    plt.savefig = savefig_with_svg
    try:
        for script in SCRIPTS:
            runpy.run_path(script, run_name="__main__")
    finally:
        plt.savefig = original_savefig

    for basename in sorted(FIGURE_BASENAMES):
        pdf_path = Path(f"{basename}.pdf")
        if pdf_path.exists():
            shutil.copy2(pdf_path, OUTPUT_DIR / pdf_path.name)

    print(f"Vector files written to: {OUTPUT_DIR.resolve()}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.suffix.lower() in {".pdf", ".svg"}:
            print(f"  - {path.name}")


if __name__ == "__main__":
    main()
