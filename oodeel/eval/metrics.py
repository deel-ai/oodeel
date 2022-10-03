import numpy as np
from ..types import *
import sklearn


def bench_metrics(
    scores: np.ndarray, 
    labels: np.ndarray, 
    metrics: Optional[List[str]] = ["auroc", "fpr95tpr"], 
    threshold: Optional[float] = None,
    step: Optional[int] = 4
) -> dict:
    """
    Compute various common metrics from OODmodel scores.
    Only AUROC for now. Also returns the 
    positive and negative mtrics curve for visualizations.

    Args:
        scores: scores output of oodmodel to evaluate
        labels: 1 if ood else 0
        step: integration step (wrt percentile). Defaults to 4.

    Returns:
        _description_
    """
    metrics_dict = {}
    fpr, tpr = get_curve(scores, labels, step)

    for metric in metrics:

        if metric == "auroc":
            auroc = -np.trapz(1.-fpr, tpr)
            metrics_dict["auroc"] = auroc

        elif metric == "fpr95tpr":
            for i, tp in enumerate(tpr):
                if tp < 0.95:
                    ind = i
                    break
            metrics_dict["fpr95tpr"] = fpr[ind]

        elif metric.__name__ in sklearn.metrics.__all__:
            if metric.__name__[:3] == "roc":
                metrics_dict[metric.__name__] = metric(labels, scores)
            else:
                if threshold is None:
                    print(f"No threshold is specified for metric {metric.__name__}, skipping")
                else:
                    oodness = [1 if x > threshold else 0 for x in scores]
                    metrics_dict[metric.__name__] = metric(labels, oodness)

        else:
            print(f"Metric {metric.__name__} not implemented, skipping")



    return metrics_dict


def get_curve(
    scores: np.ndarray, 
    labels: np.ndarray, 
    step: Optional[int] = 4, 
    return_raw: Optional[bool] = False
) -> Union[
        Tuple[
            Tuple[np.ndarray],
            Tuple[np.ndarray]
        ],
        Tuple[np.ndarray]
    ]:
    """
    Computes the number of
        * true positives,
        * false positives,
        * true negatives,
        * false negatives,
    for different threshold values. The values are uniformly 
    distributed among the percentiles, with a step = 4 / scores.shape[0]
    
    Args:
        scores: scores output of oodmodel to evaluate
        labels: 1 if ood else 0
        step: integration step (wrt percentile). Defaults to 4.

    Returns:
        4 arrays of metrics 
    """
    tpc = np.array([])
    fpc = np.array([])
    tnc = np.array([])
    fnc = np.array([])
    thresholds = np.sort(scores)
    for i in range(1, len(scores), step):
        fp, tp, fn, tn = ftpn(scores, labels, thresholds[i])
        tpc = np.append(tpc, tp)
        fpc = np.append(fpc, fp)
        tnc = np.append(tnc, tn)
        fnc = np.append(fnc, fn)

    fpr = np.concatenate([[1.], fpc/(fpc + tnc), [0.]])
    tpr = np.concatenate([[1.], tpc/(tpc + fnc), [0.]])

    if return_raw:
        return (fpc, tpc, fnc, tnc), (fpr, tpr)
    else:
        return fpr, tpr



def ftpn(
    scores: np.ndarray, 
    labels: np.ndarray, 
    threshold: float
) -> Tuple[float]:
    """
    Computes the number of
        * true positives,
        * false positives,
        * true negatives,
        * false negatives,
    for a given threshold
    
    Args:
        scores: scores output of oodmodel to evaluate
        labels: 1 if ood else 0
        threshold: 

    Returns:
        4 metrics 
    """
    pos = np.where(scores >= threshold)
    neg = np.where(scores < threshold)
    n_pos = len(pos[0])
    n_neg = len(neg[0])

    tp = np.sum(labels[pos])
    fp = n_pos - tp
    fn = np.sum(labels[neg])
    tn = n_neg - fn

    return fp, tp, fn, tn
    
    