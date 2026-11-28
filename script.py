"""
Group  1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MAT_PATH = "data/umist_cropped.mat"
RANDOM_STATE = 42
TRAIN, VAL, TEST = 0.70, 0.15, 0.15


# Loading the dataset 
def explore_mat(mat):
    """Print every variable in the .mat file: name, type, shape."""
    print("=== Structure of the .mat file ===")
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        shape = getattr(val, "shape", None)
        print(f"  {key:12s} type={type(val).__name__:10s} shape={shape} dtype={getattr(val, 'dtype', None)}")
    if "facedat" in mat:
        cells = mat["facedat"][0]
        print(f"\n'facedat' is a cell array with {len(cells)} cells (one per person).")
        print("Image array shape in each cell (H, W, n_images):")
        for i, c in enumerate(cells):
            print(f"  person {i:2d}: {c.shape}")


def load_umist(path=MAT_PATH):
    """Return images (N, H, W) as float32 and labels (N,) as ints 0..19."""
    try:
        mat = loadmat(path)
    except NotImplementedError:
        import h5py  # v7.3 files are HDF5
        with h5py.File(path, "r") as f:
            print("v7.3 file. Top-level keys:", list(f.keys()))
        raise SystemExit("Adapt this loader to the keys printed above.")

    explore_mat(mat)

    if "facedat" in mat:                       # standard UMIST layout
        images, labels = [], []
        for pid, person in enumerate(mat["facedat"][0]):   # person: (H, W, n_i)
            for k in range(person.shape[2]):
                images.append(person[:, :, k])
                labels.append(pid)
        return np.array(images, dtype=np.float32), np.array(labels)

    for xk, yk in [("X", "y"), ("data", "labels")]:       # fallback layouts
        if xk in mat and yk in mat:
            return mat[xk].astype(np.float32), mat[yk].ravel()

    raise KeyError("Unknown structure - check the printed variable names and adjust.")


def build_dataframe(images, labels):
    """One row per image: pixel columns px_0 ... px_(H*W-1) plus a 'label' column."""
    n, h, w = images.shape
    df = pd.DataFrame(images.reshape(n, h * w), columns=[f"px_{i}" for i in range(h * w)])
    df["label"] = labels
    return df, (h, w)


def plot_samples(images, labels, n_people=10, per_person=5, fname="sample_faces.png"):
    """Grid of sample faces, each row labeled with its person ID."""
    fig, axes = plt.subplots(n_people, per_person, figsize=(per_person * 1.5, n_people * 1.8))
    for p in range(n_people):
        idx = np.where(labels == p)[0]
        pick = idx[np.linspace(0, len(idx) - 1, per_person).astype(int)]  # spread across poses
        for j, ax in enumerate(axes[p]):
            ax.imshow(images[pick[j]], cmap="gray")
            ax.axis("off")
        axes[p][0].set_title(f"Person {p}", fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()


# TASK 2 - Stratified split + normalization

def stratified_split(df):
    """70/15/15 split, stratified by person so every subset has every person."""
    X, y = df.drop(columns="label"), df["label"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=VAL + TEST, stratify=y, random_state=RANDOM_STATE)

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=TEST / (VAL + TEST), stratify=y_tmp, random_state=RANDOM_STATE)

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize(X_train, X_val, X_test):
    """Fit the scaler on TRAIN only, then transform val and test (no leakage)."""
    scaler = StandardScaler()
    return (scaler.fit_transform(X_train),
            scaler.transform(X_val),
            scaler.transform(X_test),
            scaler)


def plot_distributions(y_all, y_train, y_val, y_test, fname="split_distribution.png"):
    """Bar chart of images per person in each subset + a summary table."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (name, y, c) in zip(axes, [("Train", y_train, "tab:blue"),
                                       ("Validation", y_val, "tab:orange"),
                                       ("Test", y_test, "tab:green")]):
        counts = y.value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=c)
        ax.set_title(f"{name} set - images per person (n={len(y)})")
        ax.set_xlabel("Person ID")
        ax.set_ylabel("Number of images")
        ax.set_xticks(range(len(counts)))
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

    summary = pd.DataFrame({
        "Total": y_all.value_counts().sort_index(),
        "Train": y_train.value_counts().sort_index(),
        "Val": y_val.value_counts().sort_index(),
        "Test": y_test.value_counts().sort_index(),
    })
    print("\nImages per person in each subset:\n", summary.to_string())
    summary.to_csv("split_summary.csv")


# TASK 3 - Dimensionality Reduction (PCA)

def analyze_pca_variance(X_train_s, components=(10, 20, 50, 100), fname="pca_explained_variance.png"):
    """Evaluate explained variance for multiple component counts and plot cumulative curve."""
    print("\n=== Task 3: PCA Explained Variance Analysis ===")
    for k in components:
        pca_k = PCA(n_components=k, random_state=RANDOM_STATE).fit(X_train_s)
        var_pct = np.sum(pca_k.explained_variance_ratio_) * 100
        print(f"  n_components = {k:3d} -> Cumulative Explained Variance: {var_pct:.2f}%")

    max_k = min(150, X_train_s.shape[0], X_train_s.shape[1])
    pca_full = PCA(n_components=max_k, random_state=RANDOM_STATE).fit(X_train_s)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].bar(range(1, 51), pca_full.explained_variance_ratio_[:50] * 100, color="tab:blue", alpha=0.85)
    axes[0].set_title("Individual Explained Variance (Top 50 Components)")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")

    axes[1].plot(range(1, max_k + 1), cum_var, linewidth=2, color="tab:green")
    for k, col in zip(components, ["tab:red", "tab:orange", "tab:purple", "tab:brown"]):
        if k <= max_k:
            axes[1].axvline(k, color=col, linestyle="--", alpha=0.8, label=f"{k} comps ({cum_var[k-1]:.1f}%)")
    axes[1].axhline(90, color="gray", linestyle=":", label="90% Threshold")
    axes[1].set_title("Cumulative Explained Variance Ratio (Train Set)")
    axes[1].set_xlabel("Number of Principal Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()


def plot_pca_2d(X_train_s, y_train, fname="pca_2d_projection.png"):
    """Visualize the first 2 principal components colored by Person ID."""
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_train_s)
    v1, v2 = pca_2d.explained_variance_ratio_ * 100

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap="tab20", s=30, alpha=0.85)
    plt.colorbar(sc, label="Person ID (0-19)")
    plt.title(f"2D PCA Projection of UMIST Faces (PC1: {v1:.1f}%, PC2: {v2:.1f}%)")
    plt.xlabel(f"PC 1 ({v1:.1f}% variance)")
    plt.ylabel(f"PC 2 ({v2:.1f}% variance)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()


def apply_pca(X_train_s, X_val_s, X_test_s, n_components=50):
    """Fit PCA on TRAIN only, then transform train, val, and test (no leakage)."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return (pca.fit_transform(X_train_s),
            pca.transform(X_val_s),
            pca.transform(X_test_s),
            pca)


# Main
if __name__ == "__main__":
    # ---- Task 1 ----
    images, labels = load_umist()
    print(f"\nImages: {images.shape} | Labels: {labels.shape}")
    print(f"Pixel range: {images.min():.0f} - {images.max():.0f} | People: {len(np.unique(labels))}")
    print("Images per person:\n", pd.Series(labels).value_counts().sort_index().to_string())

    df, (H, W) = build_dataframe(images, labels)
    print(f"\nDataFrame shape: {df.shape}  ({H}x{W} = {H*W} pixel columns + 'label')")
    print(df[["px_0", "px_1", "px_2", "label"]].head())

    plot_samples(images, labels)

    # ---- Task 2 ----
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)
    print(f"\nSplit sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    plot_distributions(df["label"], y_train, y_val, y_test)

    X_train_s, X_val_s, X_test_s, scaler = normalize(X_train, X_val, X_test)
    print(f"\nTrain  mean={X_train_s.mean():.3f}, std={X_train_s.std():.3f}")
    print(f"Val    mean={X_val_s.mean():.3f}, std={X_val_s.std():.3f}")
    print(f"Test   mean={X_test_s.mean():.3f}, std={X_test_s.std():.3f}")

    # Shared file so every teammate uses the identical split in Tasks 3-5
    np.savez_compressed("umist_splits.npz",
                        X_train=X_train_s, X_val=X_val_s, X_test=X_test_s,
                        y_train=y_train.values, y_val=y_val.values, y_test=y_test.values,
                        img_h=H, img_w=W)
    print("\nSaved: umist_splits.npz, split_summary.csv, sample_faces.png, split_distribution.png")

    # ---- Task 3 (PCA Dimensionality Reduction) ----
    analyze_pca_variance(X_train_s, components=[10, 20, 50, 100])
    plot_pca_2d(X_train_s, y_train.values)

    X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(X_train_s, X_val_s, X_test_s, n_components=50)
    print(f"\nPCA Reduced Shapes (n_components=50):")
    print(f"  X_train_pca={X_train_pca.shape}, X_val_pca={X_val_pca.shape}, X_test_pca={X_test_pca.shape}")

    np.savez_compressed("umist_pca_splits.npz",
                        X_train_pca=X_train_pca, X_val_pca=X_val_pca, X_test_pca=X_test_pca,
                        y_train=y_train.values, y_val=y_val.values, y_test=y_test.values)
    print("Saved: umist_pca_splits.npz, pca_explained_variance.png, pca_2d_projection.png")