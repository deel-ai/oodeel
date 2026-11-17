import numpy as np
import pytest
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve

from oodeel.eval.metrics import bench_metrics
from oodeel.methods import Forte
from oodeel.methods.forte import prdc_per_point
from tests.tests_tensorflow import generate_data_tf
from tests.tests_tensorflow import generate_model
from tests.tests_tensorflow.tf_methods_utils import load_blob_mlp
from tests.tests_tensorflow.tf_methods_utils import load_blobs_data


@pytest.mark.parametrize(
    "density_model,auroc_thr,fpr95_thr",
    [
        ("gmm", 0.95, 0.05),
        ("kde", 0.8, 0.3),
        ("ocsvm", 0.95, 0.05),
    ],
)
def test_forte(density_model, auroc_thr, fpr95_thr):
    """
    Test Forte on toy blobs OOD dataset-wise task

    We check that the area under ROC is above a certain threshold, and that the FPR95TPR
    is below an other threshold.
    """
    tf.random.set_seed(0)
    ds_fit, ds_in, ds_out = load_blobs_data()
    model = load_blob_mlp()

    forte = Forte(density_model=density_model)
    forte.fit(model, feature_layers_id=[-2], fit_dataset=ds_fit)

    scores_in, info_in = forte.score(ds_in)
    scores_out, info_out = forte.score(ds_out)
    assert scores_in.shape == (1028,)
    assert info_in["labels"].shape == (1028,)
    assert info_in["logits"].shape == (1028, 2)
    assert scores_out.shape == (972,)
    assert info_out["labels"].shape == (972,)
    assert info_out["logits"].shape == (972, 2)

    # the OOD blobs are far from the reference sample, so most of their points have
    # the same PRDC statistics (all zeros) and the same score. The fpr95tpr of
    # bench_metrics interpolates between thresholds, which is not suited to such
    # ties, so it is computed from the ROC curve instead.
    auroc = bench_metrics((scores_in, scores_out), metrics=["auroc"])["auroc"]
    labels = np.concatenate([np.zeros_like(scores_in), np.ones_like(scores_out)])
    fpr, tpr, _ = roc_curve(labels, np.concatenate([scores_in, scores_out]))
    fpr95tpr = fpr[np.argmax(tpr >= 0.95)]
    assert auroc >= auroc_thr, f"got a score of {auroc}, below {auroc_thr}!"
    assert fpr95tpr <= fpr95_thr, f"got a score of {fpr95tpr}, above {fpr95_thr}!"


def test_forte_prdc_per_point():
    """
    Test the per-point PRDC statistics against their definition
    """
    rng = np.random.RandomState(0)
    nearest = 3
    reference = rng.randn(60, 8)
    sample = rng.randn(40, 8) + 0.5

    dist = cdist(reference, sample)
    reference_radii = np.sort(cdist(reference, reference), axis=1)[:, nearest]
    sample_radii = np.sort(cdist(sample, sample), axis=1)[:, nearest]
    in_reference_balls = dist < reference_radii[:, None]
    expected = np.stack(
        [
            (dist < sample_radii[None, :]).mean(axis=0),
            in_reference_balls.sum(axis=0) / (nearest * len(reference)),
            in_reference_balls.any(axis=0),
            dist.min(axis=0) < sample_radii,
        ],
        axis=1,
    )

    stats = prdc_per_point(reference, sample, nearest=nearest)
    assert stats.shape == (40, 4)
    np.testing.assert_allclose(stats, expected)


def test_forte_dataset_as_one_sample():
    """
    Test that Forte scores a dataset as one evaluated sample, whatever its batch size,
    with several feature layers
    """
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tf(x_shape=input_shape, num_labels=num_labels, samples=samples)
    x = tf.concat([elem[0] for elem in data.batch(samples)], axis=0)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    forte = Forte()
    forte.fit(model, data.batch(50), feature_layers_id=[-5, -2])
    assert forte.aggregator is None

    scores, info = forte.score(data.batch(50))
    assert scores.shape == (100,)
    assert info["labels"].shape == (100,)
    assert info["logits"].shape == (100, 10)

    # same sample, other batch sizes
    scores_other_batch, _ = forte.score(data.batch(32))
    scores_tensor, info = forte.score(x)
    assert info["logits"].shape == (100, 10)
    np.testing.assert_allclose(scores, scores_other_batch, rtol=1e-5)
    np.testing.assert_allclose(scores, scores_tensor, rtol=1e-5)

    # a part of the dataset is another sample, with other scores
    scores_part, _ = forte.score(x[:50])
    assert scores_part.shape == (50,)
    assert not np.allclose(scores[:50], scores_part)

    # too few points to compute nearest neighbor radii
    with pytest.raises(ValueError):
        forte.score(x[:5])
