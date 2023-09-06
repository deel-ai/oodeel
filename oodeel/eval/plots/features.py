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
from matplotlib.lines import Line2D
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


def plot_2D_features(
    model: Callable,
    in_dataset: DatasetType,
    output_layer_id: Union[int, str],
    out_dataset: DatasetType = None,
    proj_method: str = "TSNE",
    max_samples: int = 4000,
    title: str = None,
    **proj_kwargs,
):
    """Visualize ID and OOD features of a model on a 2D plan using dimensionality
    reduction methods and matplotlib scatter function. Different projection methods are
    available: TSNE, PCA.

    Args:
        model (Callable): Torch or Keras model.
        in_dataset (DatasetType): In-distribution dataset (torch dataloader or tf
            dataset) that will be projected on the model feature space.
        output_layer_id (Union[int, str]): Identifier for the layer to inspect.
        out_dataset (DatasetType, optional): Out-of-distribution dataset (torch
            dataloader or tf dataset) that will be projected on the model feature space
            if not equal to None. Defaults to None.
        proj_method (str, optional): Projection method for 2d dimensionality reduction.
            Defaults to "TSNE", alternative: "PCA".
        max_samples (int, optional): Max samples to display on the scatter plot.
            Defaults to 4000.
        title (str, optional): Custom figure title. Defaults to None.
    """

    _plot_features(
        model=model,
        in_dataset=in_dataset,
        output_layer_id=output_layer_id,
        out_dataset=out_dataset,
        proj_method=proj_method,
        max_samples=max_samples,
        title=title,
        n_components=2,
        **proj_kwargs,
    )


def plot_3D_features(
    model: Callable,
    in_dataset: DatasetType,
    output_layer_id: Union[int, str],
    out_dataset: DatasetType = None,
    proj_method: str = "TSNE",
    max_samples: int = 4000,
    title: str = None,
    **proj_kwargs,
):
    """Visualize ID and OOD features of a model on a 3D space using dimensionality
    reduction methods and matplotlib scatter function. Different projection methods are
    available: TSNE, PCA.

    Args:
        model (Callable): Torch or Keras model.
        in_dataset (DatasetType): In-distribution dataset (torch dataloader or tf
            dataset) that will be projected on the model feature space.
        output_layer_id (Union[int, str]): Identifier for the layer to inspect.
        out_dataset (DatasetType, optional): Out-of-distribution dataset (torch
            dataloader or tf dataset) that will be projected on the model feature space
            if not equal to None. Defaults to None.
        proj_method (str, optional): Projection method for 2d dimensionality reduction.
            Defaults to "TSNE", alternative: "PCA".
        max_samples (int, optional): Max samples to display on the scatter plot.
            Defaults to 4000.
        title (str, optional): Custom figure title. Defaults to None.
    """
    _plot_features(
        model=model,
        in_dataset=in_dataset,
        output_layer_id=output_layer_id,
        out_dataset=out_dataset,
        proj_method=proj_method,
        max_samples=max_samples,
        title=title,
        n_components=3,
        **proj_kwargs,
    )


def _plot_features(
    model: Callable,
    in_dataset: DatasetType,
    output_layer_id: Union[int, str],
    out_dataset: DatasetType = None,
    proj_method: str = "TSNE",
    max_samples: int = 4000,
    title: str = None,
    n_components: int = 2,
    **proj_kwargs,
):
    """Visualize ID and OOD features of a model on a 2D or 3D space using dimensionality
    reduction methods and matplotlib scatter function. Different projection methods are
    available: TSNE, PCA.

    Args:
        model (Callable): Torch or Keras model.
        in_dataset (DatasetType): In-distribution dataset (torch dataloader or tf
            dataset) that will be projected on the model feature space.
        output_layer_id (Union[int, str]): Identifier for the layer to inspect.
        out_dataset (DatasetType, optional): Out-of-distribution dataset (torch
            dataloader or tf dataset) that will be projected on the model feature space
            if not equal to None. Defaults to None.
        proj_method (str, optional): Projection method for 2d dimensionality reduction.
            Defaults to "TSNE", alternative: "PCA".
        max_samples (int, optional): Max samples to display on the scatter plot.
            Defaults to 4000.
        title (str, optional): Custom figure title. If None a default one is provided.
            Defaults to None.
    """
    assert n_components in [2, 3], "The number of components should be 2 or 3"
    max_samples = max_samples if out_dataset is None else max_samples // 2

    # feature extractor
    _, _, op, FeatureExtractorClass = import_backend_specific_stuff(model)
    feature_extractor = FeatureExtractorClass(model, [output_layer_id])

    # === extract id features ===
    # features
    in_features, _ = feature_extractor.predict(in_dataset)
    in_features = op.convert_to_numpy(op.flatten(in_features))[:max_samples]

    # labels
    in_labels = []
    for _, batch_y in in_dataset:
        in_labels.append(op.convert_to_numpy(batch_y))
    in_labels = np.concatenate(in_labels)[:max_samples]
    in_labels_str = list(map(lambda x: f"class {x}", in_labels))

    # === extract ood features ===
    if out_dataset is not None:
        # features
        out_features, _ = feature_extractor.predict(out_dataset)
        out_features = op.convert_to_numpy(op.flatten(out_features))[:max_samples]

        # labels
        out_labels_str = np.array(["unknown"] * len(out_features))

        # concatenate id and ood items
        features = np.concatenate([out_features, in_features])
        labels_str = np.concatenate([out_labels_str, in_labels_str])
        data_type = np.array(
            ["OOD"] * len(out_labels_str) + ["ID"] * len(in_labels_str)
        )
    else:
        features = in_features
        labels_str = in_labels_str
        data_type = np.array(["ID"] * len(in_labels))

    # === project on 2d/3d space using tsne or pca ===
    proj_class = PROJ_DICT[proj_method]["class"]
    p_kwargs = PROJ_DICT[proj_method]["default_kwargs"]
    p_kwargs.update(proj_kwargs)
    projector = proj_class(
        n_components=n_components,
        **p_kwargs,
    )
    features_proj = projector.fit_transform(features)

    # === plot 2d/3d features ===
    features_dim = features.shape[1]
    method_str = PROJ_DICT[proj_method]["name"]
    title = (
        title
        or f"{method_str} {n_components}D projection\n"
        + f"[layer {output_layer_id}, dim: {features_dim}]"
    )

    ax = plt.axes(plt.gca())
    if n_components == 3:
        if ax.name != "3d":
            ax.remove()
            ax = plt.axes(projection="3d")
        ax.set_facecolor("white")

    # 2D projection
    if n_components == 2:
        # id data
        x, y = features_proj.T
        df = pd.DataFrame(
            {
                "dim 1": x,
                "dim 2": y,
                "Class": labels_str,
                "Data type": data_type,
            }
        )
        s = sns.scatterplot(
            data=df,
            x="dim 1",
            y="dim 2",
            hue="Class",
            hue_order=np.unique(df["Class"]),
            size="Data type",
            sizes=[40, 20],
            style="Data type",
            size_order=["ID", "OOD"],
            style_order=["ID", "OOD"],
            ax=ax,
        )
        s.legend(fontsize=8, bbox_to_anchor=(1.1, 1), borderaxespad=0)

    # 3D projection
    elif n_components == 3:
        cmap = plt.get_cmap(
            "tab10", int(np.max(in_labels)) - int(np.min(in_labels)) + 1
        )
        # id
        x_in, y_in, z_in = features_proj[len(out_features) :].T
        s = ax.scatter(
            x_in,
            y_in,
            z_in,
            c=in_labels,
            marker="D",
            label="ID data",
            s=30,
            alpha=1.0,
            cmap=cmap,
            vmin=np.min(in_labels) - 0.5,
            vmax=np.max(in_labels) + 0.5,
            edgecolors="white",
            linewidths=0.3,
        )
        # ood
        x_out, y_out, z_out = features_proj[: len(out_features)].T
        ax.scatter(
            x_out,
            y_out,
            z_out,
            c="darkslategray",
            marker="o",
            label="OOD data",
            s=15,
            alpha=1.0,
            edgecolors="white",
            linewidths=0.3,
        )
        legend_elements = [
            Line2D(
                [],
                [],
                marker="D",
                color="white",
                linestyle="None",
                label=f"class {v}",
                markerfacecolor=cmap(v),
                markersize=7,
                linewidth=0.3,
            )
            for v in np.unique(in_labels)
        ] + [
            Line2D(
                [],
                [],
                marker="o",
                color="white",
                linestyle="None",
                label="unknown",
                markerfacecolor="darkslategray",
                markersize=7,
                linewidth=0.3,
            )
        ]
        ax.legend(
            title="classes",
            handles=legend_elements,
            loc="upper right",
            fontsize=8,
            bbox_to_anchor=(1.35, 1),
            borderaxespad=0,
        )

    plt.title(title, weight="bold").set_fontsize(11)
    if n_components == 2:
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
    if n_components == 3:
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        X = np.concatenate([x_in, x_out])
        Y = np.concatenate([y_in, y_out])
        Z = np.concatenate([z_in, z_out])
        ax.set_xlim([X.min(), X.max()])
        ax.set_ylim([Y.min(), Y.max()])
        ax.set_zlim([Z.min(), Z.max()])
