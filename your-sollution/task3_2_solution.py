from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "task3_2"
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "task3_2_results"
SEED = 42
CLUSTERS = 9


def image_features(images):
    pixels = images.astype("float32") / 255
    features = []
    for level in [10, 50, 90]:
        features.append(np.percentile(pixels, level, axis=(1, 2)))
    features.append(pixels.std(axis=(1, 2)))
    for distance in [1, 3, 6]:
        horizontal = np.abs(pixels[:, :, distance:] - pixels[:, :, :-distance])
        vertical = np.abs(pixels[:, distance:] - pixels[:, :-distance])
        contrast = (horizontal.mean(axis=(1, 2)) + vertical.mean(axis=(1, 2))) / 2
        features.append(contrast)
    return np.column_stack(features)


def plot_projection(features, labels, clusters):
    pca = PCA(n_components=2)
    points = pca.fit_transform(features)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, groups, title in zip(axes, [labels, clusters], ["Tissue class", "KMeans cluster"]):
        dots = ax.scatter(points[:, 0], points[:, 1], c=groups, s=5, alpha=0.6,
                          cmap="tab10", vmin=-0.5, vmax=9.5)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    fig.colorbar(dots, ax=axes, ticks=range(CLUSTERS), label="ID")
    fig.savefig(FIGURES / "pca.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_centers(model):
    stats = ["p10", "p50", "p90", "sd", "diff1", "diff3", "diff6"]
    names = []
    for stat in stats:
        for channel in "RGB":
            names.append(f"{channel} {stat}")
    centers = pd.DataFrame(model.cluster_centers_, columns=names)
    centers.to_csv(FIGURES / "centers.csv", index_label="cluster")
    fig, ax = plt.subplots(figsize=(10, 4))
    limit = np.abs(model.cluster_centers_).max()
    heatmap = ax.imshow(model.cluster_centers_, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
    ax.set_yticks(range(CLUSTERS))
    ax.set_ylabel("Cluster")
    fig.colorbar(heatmap, ax=ax, label="Standard deviations from mean")
    fig.tight_layout()
    fig.savefig(FIGURES / "centers.png", dpi=130)
    plt.close(fig)


def plot_examples(images, clusters, distances):
    fig, axes = plt.subplots(CLUSTERS, 4, figsize=(5, 9))
    rng = np.random.default_rng(SEED)
    for group in range(CLUSTERS):
        members = np.flatnonzero(clusters == group)
        closest = members[np.argmin(distances[members, group])]
        others = members[members != closest]
        selected = [closest]
        selected.extend(rng.choice(others, min(3, len(others)), replace=False))
        for ax in axes[group]:
            ax.set_xticks([])
            ax.set_yticks([])
        for ax, index in zip(axes[group], selected):
            ax.imshow(images[index])
        axes[group, 0].set_ylabel(str(group), rotation=0, labelpad=10)
    axes[0, 0].set_title("Nearest")
    for ax in axes[0, 1:]:
        ax.set_title("Sample")
    fig.tight_layout()
    fig.savefig(FIGURES / "cluster_examples.png", dpi=120)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "train.npz") as data:
        images = data["images"]
    scaler = StandardScaler()
    features = scaler.fit_transform(image_features(images))
    model = KMeans(n_clusters=CLUSTERS, n_init=20, random_state=SEED)
    clusters = model.fit_predict(features)

    with np.load(DATA / "test_features.npz") as test:
        test_features = scaler.transform(image_features(test["images"]))
        predictions = model.predict(test_features)
        result = pd.DataFrame({"image_id": test["image_id"], "prediction": predictions})
    result.to_csv(OUTPUT / "task3_2_predictions.csv", index=False)

    with np.load(DATA / "train.npz") as data:
        labels = data["prediction"]
        names = data["label_names"]
    scores = {
        "ARI": adjusted_rand_score(labels, clusters),
        "NMI": normalized_mutual_info_score(labels, clusters),
        "silhouette": silhouette_score(features, clusters),
    }
    pd.DataFrame([scores]).to_csv(FIGURES / "metrics.csv", index=False)
    table = pd.crosstab(names[labels], clusters, rownames=["tissue"], colnames=["cluster"])
    table.loc[names].to_csv(FIGURES / "class_by_cluster.csv")
    plot_projection(features, labels, clusters)
    plot_centers(model)
    plot_examples(images, clusters, model.transform(features))
    print(scores)


if __name__ == "__main__":
    main()
