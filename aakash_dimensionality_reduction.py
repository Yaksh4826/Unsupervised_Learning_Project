"""
COMP 257 - Unsupervised Learning Final Project (Group 1)
Module: Dimensionality Reduction (PCA, Neural Autoencoder, and t-SNE)
Author: Aakash Kumar Mishra
Dataset: UMIST Cropped Face Dataset (via pre-processed umist_splits.npz)
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Reproducibility configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# UMIST cropped grayscale image dimensions (112 x 92 = 10,304 features)
IMAGE_SHAPE = (112, 92)
OUTPUT_DIR = os.path.join("outputs", "dim_reduction")


def load_preprocessed_splits(npz_path: str = "umist_splits.npz") -> tuple:
    """
    Load stratified, standardized train/validation/test splits generated
    during the data preparation stage. Automatically resolves key naming.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Could not find '{npz_path}' in {os.getcwd()}. "
            "Ensure you are running this script from the repository root."
        )

    archive = np.load(npz_path, allow_pickle=True)
    keys = set(archive.files)

    # Support both standard ('X_train') and suffixed ('X_train_scaled') naming
    if {"X_train_scaled", "X_val_scaled", "X_test_scaled"}.issubset(keys):
        X_train = archive["X_train_scaled"]
        X_val = archive["X_val_scaled"]
        X_test = archive["X_test_scaled"]
    elif {"X_train", "X_val", "X_test"}.issubset(keys):
        X_train = archive["X_train"]
        X_val = archive["X_val"]
        X_test = archive["X_test"]
    else:
        raise KeyError(f"Unexpected keys in {npz_path}: {archive.files}")

    y_train = archive["y_train"]
    y_val = archive["y_val"]
    y_test = archive["y_test"]

    print(f"Loaded '{npz_path}' successfully:")
    print(f"  Train set:      {X_train.shape} | Labels: {y_train.shape}")
    print(f"  Validation set: {X_val.shape} | Labels: {y_val.shape}")
    print(f"  Test set:       {X_test.shape} | Labels: {y_test.shape}\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    component_counts: list = [10, 20, 50, 100],
    out_dir: str = OUTPUT_DIR,
) -> dict:
    """
    Fit Principal Component Analysis (PCA) strictly on the training set
    to prevent data leakage, evaluate reconstruction fidelity across multiple
    component thresholds, and save variance and eigenface visualizations.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Fit full PCA on training data to inspect complete variance spectrum
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # 1. Cumulative Explained Variance Curve
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        linewidth=2.2,
        color="#1f77b4",
        label="Cumulative Explained Variance",
    )

    for k in component_counts:
        var_k = cumulative_variance[k - 1]
        ax.scatter(k, var_k, color="#d62728", zorder=5, s=45)
        ax.annotate(
            f"k={k} ({var_k:.1%})",
            xy=(k, var_k),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(0.90, color="#2ca02c", linestyle="--", linewidth=1.1, label="90% Threshold")
    ax.axhline(0.95, color="#555555", linestyle=":", linewidth=1.1, label="95% Threshold")
    ax.set_title("PCA Cumulative Explained Variance Ratio (Training Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Principal Components (k)", fontsize=10)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=10)
    ax.set_ylim(0.4, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_explained_variance.png"), dpi=300)
    plt.close(fig)

    # 2. Component Sweep & Visual Face Reconstructions
    pca_results = {}
    sample_idx = 0

    fig, axes = plt.subplots(1, len(component_counts) + 1, figsize=(13, 3.2))
    axes[0].imshow(X_test[sample_idx].reshape(IMAGE_SHAPE), cmap="gray")
    axes[0].set_title("Original\n(10,304 dims)", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for idx, k in enumerate(component_counts, start=1):
        pca_k = PCA(n_components=k, random_state=RANDOM_STATE)
        Z_train = pca_k.fit_transform(X_train)
        Z_val = pca_k.transform(X_val)
        Z_test = pca_k.transform(X_test)

        X_test_recon = pca_k.inverse_transform(Z_test)
        test_mse = mean_squared_error(X_test, X_test_recon)
        var_retained = float(np.sum(pca_k.explained_variance_ratio_))

        pca_results[k] = {
            "model": pca_k,
            "X_train": Z_train,
            "X_val": Z_val,
            "X_test": Z_test,
            "X_test_recon": X_test_recon,
            "test_mse": test_mse,
            "explained_variance": var_retained,
        }

        axes[idx].imshow(X_test_recon[sample_idx].reshape(IMAGE_SHAPE), cmap="gray")
        axes[idx].set_title(f"PCA k={k}\n({var_retained:.1%} var)", fontsize=9)
        axes[idx].axis("off")

    plt.suptitle("PCA Test Image Reconstruction Across Component Counts", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_reconstructions.png"), dpi=300)
    plt.close(fig)

    # 3. Top 10 Eigenfaces Visualization
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
    for i, ax in enumerate(axes.flat):
        eigenface = pca_full.components_[i].reshape(IMAGE_SHAPE)
        var_pct = pca_full.explained_variance_ratio_[i] * 100
        ax.imshow(eigenface, cmap="gray")
        ax.set_title(f"PC {i + 1} ({var_pct:.1f}%)", fontsize=9)
        ax.axis("off")

    plt.suptitle("Top 10 Principal Components (Eigenfaces)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_eigenfaces.png"), dpi=300)
    plt.close(fig)

    return pca_results


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    latent_dim: int = 50,
    out_dir: str = OUTPUT_DIR,
) -> dict:
    """
    Build and train a symmetric neural network Autoencoder in Keras to learn
    a non-linear compressed latent representation of dimension `latent_dim`.
    """
    input_dim = X_train.shape[1]

    # Encoder
    inputs = layers.Input(shape=(input_dim,), name="image_input")
    x = layers.Dense(512, name="enc_dense_1")(inputs)
    x = layers.BatchNormalization(name="enc_bn_1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="enc_act_1")(x)
    x = layers.Dropout(0.2, name="enc_dropout")(x)

    x = layers.Dense(128, name="enc_dense_2")(x)
    x = layers.BatchNormalization(name="enc_bn_2")(x)
    x = layers.LeakyReLU(alpha=0.1, name="enc_act_2")(x)

    latent = layers.Dense(latent_dim, name="latent_bottleneck")(x)

    # Decoder
    x = layers.Dense(128, name="dec_dense_1")(latent)
    x = layers.BatchNormalization(name="dec_bn_1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="dec_act_1")(x)

    x = layers.Dense(512, name="dec_dense_2")(x)
    x = layers.BatchNormalization(name="dec_bn_2")(x)
    x = layers.LeakyReLU(alpha=0.1, name="dec_act_2")(x)

    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction_output")(x)

    autoencoder = models.Model(inputs, outputs, name="UMIST_Autoencoder")
    encoder = models.Model(inputs, latent, name="UMIST_Encoder")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )

    print(f"Training Neural Autoencoder (bottleneck = {latent_dim} dims)...")
    history = autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
    )

    # 1. Plot Training & Validation MSE Loss Curve
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(history.history["loss"], linewidth=2, label="Training MSE")
    ax.plot(history.history["val_loss"], linewidth=2, label="Validation MSE")
    ax.set_title(f"Autoencoder Learning Curve (Latent Dim = {latent_dim})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "autoencoder_training_curve.png"), dpi=300)
    plt.close(fig)

    Z_train = encoder.predict(X_train, verbose=0)
    Z_val = encoder.predict(X_val, verbose=0)
    Z_test = encoder.predict(X_test, verbose=0)
    X_test_recon = autoencoder.predict(X_test, verbose=0)
    test_mse = mean_squared_error(X_test, X_test_recon)

    return {
        "autoencoder": autoencoder,
        "encoder": encoder,
        "X_train": Z_train,
        "X_val": Z_val,
        "X_test": Z_test,
        "X_test_recon": X_test_recon,
        "test_mse": test_mse,
    }


def save_comparison_visuals(
    X_test: np.ndarray,
    pca_50_res: dict,
    ae_50_res: dict,
    y_train: np.ndarray,
    out_dir: str = OUTPUT_DIR,
) -> None:
    """
    Generate side-by-side visual comparisons between linear PCA (k=50)
    and non-linear Autoencoder (latent_dim=50):
      1. Direct test-set image reconstruction comparison
      2. 2D t-SNE manifold projection colored by subject identity
    """
    # 1. Side-by-side Reconstruction Comparison (3 test subjects)
    sample_indices = [0, 15, 30]
    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(8, 7.5))
    col_titles = [
        "Original Image",
        f"PCA (k=50)\nTest MSE: {pca_50_res['test_mse']:.4f}",
        f"Autoencoder (dim=50)\nTest MSE: {ae_50_res['test_mse']:.4f}",
    ]

    for row, idx in enumerate(sample_indices):
        imgs = [
            X_test[idx].reshape(IMAGE_SHAPE),
            pca_50_res["X_test_recon"][idx].reshape(IMAGE_SHAPE),
            ae_50_res["X_test_recon"][idx].reshape(IMAGE_SHAPE),
        ]
        for col in range(3):
            axes[row, col].imshow(imgs[col], cmap="gray")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10, fontweight="bold")
            axes[row, col].axis("off")

    plt.suptitle("Linear (PCA) vs. Non-Linear (Autoencoder) Reconstruction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_vs_autoencoder_reconstructions.png"), dpi=300)
    plt.close(fig)

    # 2. 2D t-SNE Visualization of 50D Embeddings
    tsne_pca = TSNE(n_components=2, perplexity=25, random_state=RANDOM_STATE).fit_transform(
        pca_50_res["X_train"]
    )
    tsne_ae = TSNE(n_components=2, perplexity=25, random_state=RANDOM_STATE).fit_transform(
        ae_50_res["X_train"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    sc1 = axes[0].scatter(tsne_pca[:, 0], tsne_pca[:, 1], c=y_train, cmap="tab20", s=28, alpha=0.88)
    axes[0].set_title("2D t-SNE Projection of PCA (50 Components)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].grid(True, alpha=0.2)

    sc2 = axes[1].scatter(tsne_ae[:, 0], tsne_ae[:, 1], c=y_train, cmap="tab20", s=28, alpha=0.88)
    axes[1].set_title("2D t-SNE Projection of Autoencoder (50 Latent Dims)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, alpha=0.2)

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label("Subject / Person ID (0–19)", fontsize=10)
    plt.savefig(os.path.join(out_dir, "tsne_manifold_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_splits("umist_splits.npz")

    # Step 1: Linear Dimensionality Reduction via PCA
    pca_results = evaluate_pca(
        X_train, X_val, X_test, component_counts=[10, 20, 50, 100], out_dir=OUTPUT_DIR
    )

    # Step 2: Non-Linear Dimensionality Reduction via Neural Autoencoder
    ae_results = train_autoencoder(
        X_train, X_val, X_test, latent_dim=50, out_dir=OUTPUT_DIR
    )

    # Step 3: Comparative Visualizations (PCA vs Autoencoder + t-SNE)
    save_comparison_visuals(X_test, pca_results[50], ae_results, y_train, out_dir=OUTPUT_DIR)

    # Step 4: Print Quantitative Summary Table
    print("\n" + "=" * 68)
    print("DIMENSIONALITY REDUCTION BENCHMARK SUMMARY (TEST SET)")
    print("=" * 68)
    print(f"{'Method':<28} | {'Dimensions':<10} | {'Explained Var':<13} | {'Test MSE':<10}")
    print("-" * 68)
    for k, res in pca_results.items():
        print(
            f"{'PCA (Linear)':<28} | {k:<10d} | {res['explained_variance']*100:>11.2f}% | {res['test_mse']:<10.4f}"
        )
    print(
        f"{'Autoencoder (Non-Linear)':<28} | {50:<10d} | {'N/A (Non-lin)':<13} | {ae_results['test_mse']:<10.4f}"
    )
    print("=" * 68 + "\n")

    # Step 5: Export Reduced Feature Sets for Downstream Clustering & Classifier
    export_path = os.path.join(OUTPUT_DIR, "umist_reduced_features.npz")
    np.savez(
        export_path,
        X_train_pca50=pca_results[50]["X_train"],
        X_val_pca50=pca_results[50]["X_val"],
        X_test_pca50=pca_results[50]["X_test"],
        X_train_pca100=pca_results[100]["X_train"],
        X_val_pca100=pca_results[100]["X_val"],
        X_test_pca100=pca_results[100]["X_test"],
        X_train_ae50=ae_results["X_train"],
        X_val_ae50=ae_results["X_val"],
        X_test_ae50=ae_results["X_test"],
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    print(f"All visualizations saved to: '{OUTPUT_DIR}/'")
    print(f"Reduced datasets exported to: '{export_path}'")


if __name__ == "__main__":
    main()