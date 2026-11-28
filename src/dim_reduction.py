"""Run section 2.2 and write figures, diagnostics and PCA scores."""
import argparse
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
UNSCALED = "clean_bookings_unscaled.csv"
EXCLUDED = {
    "is_canceled", "reservation_status", "reservation_status_date",
    "assigned_room_type", "booking_changes", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests",
}


def read_bookings(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive, archive.open(UNSCALED) as csv:
            return pd.read_csv(csv)
    return pd.read_csv(path)


def prepare(bookings):
    """Use the encoded 2.1 columns and fit the scaler on train."""
    columns = [name for name in bookings if name not in EXCLUDED
               and not name.startswith("assigned_room_type_")]
    features = bookings[columns]
    target = bookings["is_canceled"]
    if set(target.unique()) != {0, 1} or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Expected finite encoded features and binary is_canceled.")
    train, validation = train_test_split(
        np.arange(len(bookings)), test_size=0.2, stratify=target,
        random_state=SEED,
    )
    scaler = StandardScaler().fit(features.iloc[train])
    return (
        scaler.transform(features.iloc[train]),
        scaler.transform(features.iloc[validation]),
        target.to_numpy(), train, validation, columns,
    )


def score_summary(train_scores, validation_scores, train_y, validation_y, names):
    """Fix axis orientation on train for AUC; estimate separation on validation."""
    rows = []
    for name, trained, held_out in zip(names, train_scores.T, validation_scores.T):
        direction = 1 if trained[train_y == 1].mean() >= trained[train_y == 0].mean() else -1
        kept, canceled = held_out[validation_y == 0], held_out[validation_y == 1]
        rows.append({
            "axis": name,
            "mean_0": kept.mean(),
            "mean_1": canceled.mean(),
            "std_0": kept.std(ddof=1),
            "std_1": canceled.std(ddof=1),
            "point_biserial_r": np.corrcoef(held_out, validation_y)[0, 1],
            "fisher_ratio": (canceled.mean() - kept.mean()) ** 2 /
                            (canceled.var(ddof=1) + kept.var(ddof=1)),
            "auc_oriented_from_train": roc_auc_score(validation_y, direction * held_out),
        })
    return pd.DataFrame(rows)


def save_figures(variance, scores, labels, lda_scores, folder):
    counts = np.arange(1, len(variance) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(counts, variance.cumsum() * 100)
    for level in (80, 90):
        ax.axhline(level, linestyle="--", color="gray", linewidth=0.8)
    ax.set(xlabel="Number of PCs", ylabel="Cumulative explained variance (%)", ylim=(0, 103))
    fig.tight_layout()
    fig.savefig(folder / "pca_variance.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (i, j) in zip(axes.flat, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        for label, color in [(0, "#27698c"), (1, "#bd492d")]:
            points = scores[labels == label]
            ax.scatter(points[:, i], points[:, j], s=5, alpha=0.22, color=color,
                       label=f"is_canceled = {label}")
        ax.set(xlabel=f"PC{i + 1}", ylabel=f"PC{j + 1}")
    axes.flat[0].legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(folder / "pca_pairs.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.histogram_bin_edges(lda_scores, bins=55)
    for label, color in [(0, "#27698c"), (1, "#bd492d")]:
        ax.hist(lda_scores[labels == label], bins=bins, density=True, histtype="step",
                linewidth=1.6, color=color, label=f"is_canceled = {label}")
    ax.set(xlabel="LD1", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "lda_distribution.png", dpi=160)
    plt.close(fig)


def run(path):
    bookings = read_bookings(path)
    print(f"Section 2.1 input shape: {bookings.shape}")
    tables = ROOT / "results/tables/02_dim_reduction"
    figures = ROOT / "results/figures/02_dim_reduction"
    for folder in (tables, figures):
        folder.mkdir(parents=True, exist_ok=True)

    x_train, x_val, y, train, val, names = prepare(bookings)
    y_train, y_val = y[train], y[val]
    with threadpool_limits(limits=4):
        pca = PCA(svd_solver="full").fit(x_train)
        train_pca, val_pca = pca.transform(x_train), pca.transform(x_val)
        lda = LinearDiscriminantAnalysis(n_components=1).fit(x_train, y_train)
        train_lda, val_lda = lda.transform(x_train), lda.transform(x_val)

    ratio = pca.explained_variance_ratio_
    cumulative = ratio.cumsum()
    cutoffs = {level: int(np.searchsorted(cumulative, level / 100) + 1)
               for level in (80, 90, 95)}
    axes = [f"PC{i}" for i in range(1, 5)] + ["LD1"]
    summary = score_summary(
        np.column_stack((train_pca[:, :4], train_lda)),
        np.column_stack((val_pca[:, :4], val_lda)), y_train, y_val, axes,
    )
    summary.to_csv(tables / "axis_validation.csv", index=False)
    pd.DataFrame({"pc": np.arange(1, len(ratio) + 1),
                  "eigenvalue": pca.explained_variance_,
                  "variance_ratio": ratio, "cumulative_ratio": cumulative}
                 ).to_csv(tables / "pca_variance.csv", index=False)
    pd.DataFrame(pca.components_[:4].T, index=names, columns=axes[:4]
                 ).to_csv(tables / "pca_weights.csv", index_label="feature")
    pd.DataFrame({"mean_train": x_train.mean(axis=0), "std_train": x_train.std(axis=0)},
                 index=names).to_csv(tables / "input_stats.csv", index_label="feature")
    bookings.iloc[train][names].describe().T.to_csv(tables / "unscaled_train.csv", index_label="feature")
    pca_summary = pd.DataFrame({"threshold": cutoffs.keys(), "components": cutoffs.values()})
    pca_summary.to_csv(tables / "pca_cutoffs.csv", index=False)

    # row_id is the zero-based position in the cleaned, unscaled file.
    # Keep labels separate so clustering can choose k using features alone.
    rows = np.r_[train, val]
    split = np.repeat(["train", "validation"], [len(train), len(val)])
    np.savez_compressed(tables / "pca_scores.npz", scores=np.vstack((train_pca, val_pca))[:, :cutoffs[90]],
                        row_id=rows, split=split, components_80=cutoffs[80])
    pd.DataFrame({"row_id": rows, "split": split, "is_canceled": y[rows]}
                 ).to_csv(tables / "split_labels.csv", index=False)
    save_figures(ratio, val_pca, y_val, val_lda.ravel(), figures)

    print(f"Train/validation: {x_train.shape} / {x_val.shape}")
    print(f"Canceled train/validation: {y_train.sum()} / {y_val.sum()}")
    print("PCA component counts:", cutoffs)
    print(f"Top four PCA cumulative variance: {cumulative[3]:.4%}")
    print(f"LDA relative discriminant strength: {lda.explained_variance_ratio_[0]:.4%}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Unscaled CSV from section 2.1 or its ZIP")
    run(parser.parse_args().input)
