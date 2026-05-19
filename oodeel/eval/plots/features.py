# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.lines import Line2D
from packaging.version import parse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ...types import Callable
from ...types import DatasetType
from ...types import Union
from ...utils import import_backend_specific_stuff

sns.set_style("darkgrid")

PROJ_DICT = {
    "TSNE": {
        "name": "t-SNE",
        "class": TSNE,
        "default_kwargs": dict(perplexity=30.0, n_iter=800, random_state=0),
    },
    "PCA": {"name": "PCA", "class": PCA, "default_kwargs": dict()},
}

# check sklearn version: if > 1.5, use max_iter instead of n_iter
if parse(sklearn.__version__) >= parse("1.5"):
    n_iter = PROJ_DICT["TSNE"]["default_kwargs"].pop("n_iter")
    PROJ_DICT["TSNE"]["default_kwargs"]["max_iter"] = n_iter


def _extract_features_and_labels(feature_extractor, dataset, op, max_samples=None):
    """Extract features and labels from a dataset using a feature extractor.

    Args:
        feature_extractor: The feature extractor to use.
        dataset: The dataset to extract features from.
        op: Backend operator for numpy conversion.
        max_samples (int, optional): Max samples to extract. Defaults to None.

    Returns:
        tuple: (features, labels) as numpy arrays.
    """
    features, _ = feature_extractor.predict(dataset, numpy_concat=True)
    features = features[0].reshape(features[0].shape[0], -1)

    labels = np.concatenate([op.convert_to_numpy(y) for _, y in dataset])

    if max_samples is not None:
        features, labels = features[:max_samples], labels[:max_samples]

    return features, labels


def _build_legend_handle(marker, label, color, markersize=7):
    """Create a legend handle with consistent styling."""
    return Line2D(
        [],
        [],
        marker=marker,
        color="white",
        linestyle="None",
        label=label,
        markerfacecolor=color,
        markersize=markersize,
    )


def plot_2D_features(
    model: Callable,
    in_dataset: DatasetType,
    output_layer_id: Union[int, str],
    out_dataset: DatasetType = None,
    proj_method: str = "TSNE",
    max_samples_in: int = None,
    max_samples_out: int = None,
    ood_marker_size: int = 20,
    id_marker_size: int = 20,
    ood_on_top: bool = True,
    title: str = None,
    **proj_kwargs,
):
    """Visualize ID and OOD features on a 2D plan using dimensionality reduction.

    Note: For PCA, projection is fitted on ID data only. For TSNE, all data is used
    (algorithm limitation).

    Args:
        model (Callable): Torch or Keras model.
        in_dataset (DatasetType): In-distribution dataset.
        output_layer_id (Union[int, str]): Identifier for the layer to inspect.
        out_dataset (DatasetType, optional): Out-of-distribution dataset.
        proj_method (str, optional): "TSNE" or "PCA". Defaults to "TSNE".
        max_samples_in (int, optional): Max ID samples. Defaults to None (all).
        max_samples_out (int, optional): Max OOD samples. Defaults to None (all).
        ood_marker_size (int, optional): OOD marker size. Defaults to 40.
        id_marker_size (int, optional): ID marker size. Defaults to 20.
        ood_on_top (bool, optional): Render OOD on top. Defaults to True.
        title (str, optional): Custom figure title.
    """
    _plot_features(
        model,
        in_dataset,
        output_layer_id,
        out_dataset,
        proj_method,
        max_samples_in,
        max_samples_out,
        ood_marker_size,
        id_marker_size,
        ood_on_top,
        title,
        n_components=2,
        **proj_kwargs,
    )


def plot_3D_features(
    model: Callable,
    in_dataset: DatasetType,
    output_layer_id: Union[int, str],
    out_dataset: DatasetType = None,
    proj_method: str = "TSNE",
    max_samples_in: int = None,
    max_samples_out: int = None,
    ood_marker_size: int = 30,
    id_marker_size: int = 15,
    ood_on_top: bool = True,
    title: str = None,
    **proj_kwargs,
):
    """Visualize ID and OOD features in 3D space using dimensionality reduction.

    Note: For PCA, projection is fitted on ID data only. For TSNE, all data is used
    (algorithm limitation).

    Args:
        model (Callable): Torch or Keras model.
        in_dataset (DatasetType): In-distribution dataset.
        output_layer_id (Union[int, str]): Identifier for the layer to inspect.
        out_dataset (DatasetType, optional): Out-of-distribution dataset.
        proj_method (str, optional): "TSNE" or "PCA". Defaults to "TSNE".
        max_samples_in (int, optional): Max ID samples. Defaults to None (all).
        max_samples_out (int, optional): Max OOD samples. Defaults to None (all).
        ood_marker_size (int, optional): OOD marker size. Defaults to 30.
        id_marker_size (int, optional): ID marker size. Defaults to 15.
        ood_on_top (bool, optional): Render OOD on top. Defaults to True.
        title (str, optional): Custom figure title.
    """
    _plot_features(
        model,
        in_dataset,
        output_layer_id,
        out_dataset,
        proj_method,
        max_samples_in,
        max_samples_out,
        ood_marker_size,
        id_marker_size,
        ood_on_top,
        title,
        n_components=3,
        **proj_kwargs,
    )


def _plot_features(
    model: Callable,
    in_dataset: DatasetType,
    output_layer_id: Union[int, str],
    out_dataset: DatasetType = None,
    proj_method: str = "TSNE",
    max_samples_in: int = None,
    max_samples_out: int = None,
    ood_marker_size: int = 20,
    id_marker_size: int = 20,
    ood_on_top: bool = True,
    title: str = None,
    n_components: int = 2,
    **proj_kwargs,
):
    """Internal function to plot features in 2D or 3D."""
    assert n_components in [2, 3], "n_components must be 2 or 3"

    # === Extract features ===
    _, _, op, FeatureExtractorClass = import_backend_specific_stuff(model)
    feature_extractor = FeatureExtractorClass(model, [output_layer_id])

    in_features, in_labels = _extract_features_and_labels(
        feature_extractor, in_dataset, op, max_samples_in
    )

    has_ood = out_dataset is not None
    if has_ood:
        out_features, _ = _extract_features_and_labels(
            feature_extractor, out_dataset, op, max_samples_out
        )
    else:
        out_features = None

    # === Project features ===
    proj_class = PROJ_DICT[proj_method]["class"]
    p_kwargs = {**PROJ_DICT[proj_method]["default_kwargs"], **proj_kwargs}
    projector = proj_class(n_components=n_components, **p_kwargs)

    # PCA: fit on ID only; TSNE: must fit on all data (no transform method)
    if proj_method == "PCA":
        in_proj = projector.fit_transform(in_features)
        out_proj = projector.transform(out_features) if has_ood else None
    else:
        if has_ood:
            all_proj = projector.fit_transform(
                np.concatenate([in_features, out_features])
            )
            in_proj, out_proj = (
                all_proj[: len(in_features)],
                all_proj[len(in_features) :],
            )
        else:
            in_proj = projector.fit_transform(in_features)
            out_proj = None

    # === Setup plot ===
    method_str = PROJ_DICT[proj_method]["name"]
    title = title or (
        f"{method_str} {n_components}D projection\n"
        f"[layer {output_layer_id}, dim: {in_features.shape[1]}]"
    )

    ax = plt.gca()
    if n_components == 3 and ax.name != "3d":
        ax.remove()
        ax = plt.axes(projection="3d")
        ax.set_facecolor("white")

    id_zorder, ood_zorder = (1, 2) if ood_on_top else (2, 1)

    # === Plot ===
    if n_components == 2:
        _plot_2d(
            ax,
            in_proj,
            in_labels,
            out_proj,
            id_marker_size,
            ood_marker_size,
            id_zorder,
            ood_zorder,
        )
    else:
        _plot_3d(
            ax,
            in_proj,
            in_labels,
            out_proj,
            id_marker_size,
            ood_marker_size,
            id_zorder,
            ood_zorder,
        )

    plt.title(title, weight="bold", fontsize=11)


def _plot_2d(
    ax, in_proj, in_labels, out_proj, id_size, ood_size, id_zorder, ood_zorder
):
    """Plot 2D scatter with ID and optional OOD data."""
    has_ood = out_proj is not None
    in_labels_str = [f"class {x}" for x in in_labels]
    unique_classes = sorted(set(in_labels_str))

    # Plot ID
    df_in = pd.DataFrame(
        {"dim 1": in_proj[:, 0], "dim 2": in_proj[:, 1], "Class": in_labels_str}
    )
    sns.scatterplot(
        data=df_in,
        x="dim 1",
        y="dim 2",
        hue="Class",
        hue_order=unique_classes,
        s=id_size,
        marker="D",
        ax=ax,
        zorder=id_zorder,
        legend=False,
    )

    # Plot OOD
    if has_ood:
        ax.scatter(
            out_proj[:, 0],
            out_proj[:, 1],
            c="darkslategray",
            edgecolors="white",
            linewidths=0.3,
            s=ood_size,
            marker="o",
            zorder=ood_zorder,
        )

    # Build legend
    palette = sns.color_palette()
    handles = (
        [_build_legend_handle(None, "Class", "none")]
        + [
            _build_legend_handle("s", cls, palette[i % len(palette)])
            for i, cls in enumerate(unique_classes)
        ]
        + [
            _build_legend_handle(None, "", "none"),
            _build_legend_handle(None, "Data type", "none"),
            _build_legend_handle("D", "ID", "gray"),
        ]
    )
    if has_ood:
        handles.append(_build_legend_handle("o", "OOD", "darkslategray"))

    ax.legend(handles=handles, fontsize=8, bbox_to_anchor=(1.1, 1), borderaxespad=0)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")


def _plot_3d(
    ax, in_proj, in_labels, out_proj, id_size, ood_size, id_zorder, ood_zorder
):
    """Plot 3D scatter with ID and optional OOD data."""
    has_ood = out_proj is not None
    cmap = plt.get_cmap("tab10", int(np.max(in_labels)) - int(np.min(in_labels)) + 1)

    # Plot ID
    ax.scatter(
        *in_proj.T,
        c=in_labels,
        marker="D",
        s=id_size,
        cmap=cmap,
        vmin=np.min(in_labels) - 0.5,
        vmax=np.max(in_labels) + 0.5,
        edgecolors="white",
        linewidths=0.3,
        zorder=id_zorder,
    )

    # Plot OOD
    if has_ood:
        ax.scatter(
            *out_proj.T,
            c="darkslategray",
            marker="o",
            s=ood_size,
            edgecolors="white",
            linewidths=0.3,
            zorder=ood_zorder,
        )

    # Build legend
    handles = [
        _build_legend_handle("D", f"class {v}", cmap(v)) for v in np.unique(in_labels)
    ]
    handles.append(_build_legend_handle("o", "unknown", "darkslategray"))

    ax.legend(
        title="classes",
        handles=handles,
        loc="upper right",
        fontsize=8,
        bbox_to_anchor=(1.35, 1),
        borderaxespad=0,
    )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    # Set axis limits
    all_data = np.vstack([in_proj, out_proj]) if has_ood else in_proj
    ax.set_xlim([all_data[:, 0].min(), all_data[:, 0].max()])
    ax.set_ylim([all_data[:, 1].min(), all_data[:, 1].max()])
    ax.set_zlim([all_data[:, 2].min(), all_data[:, 2].max()])
