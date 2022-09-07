import numpy as np

def bench_metrics(curves, metrics=None):
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

    _, (tpr, fpr) = curves
    auroc = -np.trapz(1.-fpr, tpr)
    return (auroc)


def get_curve(scores, labels, step = 4):
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
        tp, fp, tn, fn = ftpn(scores, labels, thresholds[i])
        tpc = np.append(tpc, tp)
        fpc = np.append(fpc, fp)
        tnc = np.append(tnc, tn)
        fnc = np.append(fnc, fn)
    tpr = np.concatenate([[1.], tpc/(tpc + fnc), [0.]])
    fpr = np.concatenate([[1.], fpc/(fpc + tnc), [0.]])
    return (tpc, fpc, tnc, fnc), (tpr, fpr)



def ftpn(scores, labels, threshold):
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

    return tp, fp, tn, fn
    
    