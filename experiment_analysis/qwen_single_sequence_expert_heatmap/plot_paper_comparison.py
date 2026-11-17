#!/usr/bin/env python3
"""Replot the saved comparison at paper size; leave CSV/JSON data untouched."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    source = root / "qwen_high_low_layer47_lag128_probe245_probe237_matched_clean"
    inputs = [source.with_suffix(ext) for ext in (".csv", ".json")]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    metadata = json.loads(inputs[1].read_text())
    with inputs[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    shares, differences, tvs = [], [], []
    for label in ("high", "low"):
        group = [row for row in rows if row["drift_label"] == label]
        assert [int(row["expert_id"]) for row in group] == list(range(128))
        values = np.array([[float(row[key]) for row in group] for key in
                           ("rollout_share_pct", "training_share_pct")])
        delta = np.array([float(row["training_minus_rollout_pp"]) for row in group])
        np.testing.assert_allclose(values.sum(axis=1), 100, rtol=0, atol=1e-12)
        np.testing.assert_allclose(delta, values[1] - values[0], rtol=0, atol=1e-12)
        tv = np.abs(delta).sum() / 200
        np.testing.assert_allclose(tv, metadata[label]["raw_normalized_tv"], rtol=0, atol=1e-14)
        shares.append(values)
        differences.append(delta)
        tvs.append(tv)
    palette = metadata["palette"]
    cmap = LinearSegmentedColormap.from_list("share", palette["sequential"], N=256)
    dcmap = LinearSegmentedColormap.from_list("change", palette["difference"], N=257)
    share_norm = Normalize(*palette["range_pct"])
    change_norm = Normalize(*palette["range_pp"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#0b0b0b",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(8, 3.6), facecolor="white")
    # Explicit physical layout preserves an exact eight-inch PDF page.
    left, right, width = .115, .605, .37
    for i, (label, bottom) in enumerate(zip(("High drift", "Low drift"), (.70, .36))):
        fig.text(left, bottom + .195, label,
                 fontsize=14, weight="semibold")
        ax = fig.add_axes([left, bottom, width, .17])
        dax = fig.add_axes([right, bottom, width, .17])
        image = ax.imshow(shares[i], cmap=cmap, norm=share_norm,
                          aspect="auto", interpolation="nearest")
        ax.axhline(.5, color="white", linewidth=1.2)
        ax.set_yticks([0, 1], ["Rollout", "Training"], fontsize=12)
        dim = dax.imshow(differences[i][None, :], cmap=dcmap, norm=change_norm,
                         aspect="auto", interpolation="nearest")
        dax.set_yticks([0], ["Training\n− Rollout"], fontsize=11)
        for axis in (ax, dax):
            axis.set_xticks([0, 32, 64, 96, 127])
            axis.tick_params(axis="x", length=0, pad=4, labelsize=11)
            axis.tick_params(axis="y", length=0, pad=6)
            if i == 1:
                axis.set_xlabel("Expert ID", fontsize=12, labelpad=5)
            for spine in axis.spines.values():
                spine.set_visible(False)
    for artist, x, ticks, label in (
        (image, left, [0, 2, 4, 6, 8], "Expert selection share (%)"),
        (dim, right, [-3, -1.5, 0, 1.5, 3], "Training − rollout (pp)"),
    ):
        cb = fig.colorbar(artist, cax=fig.add_axes([x, .145, width, .035]),
                          orientation="horizontal", ticks=ticks)
        cb.set_label(label, fontsize=11, labelpad=5)
        cb.ax.tick_params(length=0, labelsize=11, pad=3)
        cb.outline.set_visible(False)
    output = root / "高低漂移对比"
    for ext in (".png", ".pdf"):
        fig.savefig(output.with_suffix(ext), dpi=300, facecolor="white")
    plt.close(fig)
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    print(json.dumps({"tv": dict(zip(("high", "low"), tvs)),
                      "size_inches": [8, 3.6], "font_pt": [11, 12, 14],
                      "range_pct": palette["range_pct"], "range_pp": palette["range_pp"],
                      "inputs_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
