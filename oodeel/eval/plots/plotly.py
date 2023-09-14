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
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
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


def plotly_3D_features(
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
    available: TSNE, PCA. This function requires the package plotly to be installed to
    run an interactive 3D scatter plot.

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
    max_samples = max_samples if out_dataset is None else max_samples // 2

    # feature extractor
    _, _, op, FeatureExtractorClass = import_backend_specific_stuff(model)
    feature_extractor = FeatureExtractorClass(model, [output_layer_id])

    # === extract id features ===
    # features
    in_features, _ = feature_extractor.predict(in_dataset)
    in_features = op.convert_to_numpy(op.flatten(in_features[0]))[:max_samples]

    # labels
    in_labels = []
    for _, batch_y in in_dataset:
        in_labels.append(op.convert_to_numpy(batch_y))
    in_labels = np.concatenate(in_labels)[:max_samples]
    in_labels = list(map(lambda x: f"class {x}", in_labels))

    # === extract ood features ===
    if out_dataset is not None:
        # features
        out_features, _ = feature_extractor.predict(out_dataset)
        out_features = op.convert_to_numpy(op.flatten(out_features[0]))[:max_samples]

        # labels
        out_labels = np.array(["unknown"] * len(out_features))

        # concatenate id and ood items
        features = np.concatenate([out_features, in_features])
        labels = np.concatenate([out_labels, in_labels])
        data_type = np.array(["OOD"] * len(out_labels) + ["ID"] * len(in_labels))
        points_size = np.array([1] * len(out_labels) + [3] * len(in_labels))
    else:
        features = in_features
        labels = in_labels
        data_type = np.array(["ID"] * len(in_labels))
        points_size = np.array([3] * len(in_labels))

    # === project on 3d space using tsne or pca ===
    proj_class = PROJ_DICT[proj_method]["class"]
    p_kwargs = PROJ_DICT[proj_method]["default_kwargs"]
    p_kwargs.update(proj_kwargs)
    projector = proj_class(
        n_components=3,
        **p_kwargs,
    )
    features_proj = projector.fit_transform(features)

    # === plot 3d features ===
    features_dim = features.shape[1]
    method_str = PROJ_DICT[proj_method]["name"]
    title = (
        title
        or f"{method_str} 3D projection\n"
        + f"[layer {output_layer_id}, dim: {features_dim}]"
    )

    x, y, z = features_proj.T
    df = pd.DataFrame(
        {
            "dim 1": x,
            "dim 2": y,
            "dim 3": z,
            "class": labels,
            "data type": data_type,
            "size": points_size,
        }
    )

    # 3D projection
    fig = px.scatter_3d(
        data_frame=df,
        x="dim 1",
        y="dim 2",
        z="dim 3",
        color="class",
        symbol="data type",
        size="size",
        opacity=1,
        category_orders={"class": np.unique(df["class"])},
        symbol_map={"OOD": "circle", "ID": "diamond"},
    )

    fig.update_layout(
        title={"text": title, "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
    )
    fig.show()
