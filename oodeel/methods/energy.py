import tensorflow as tf
from .base import OODModel
import numpy as np
from ..types import *
from scipy.special import logsumexp


class Energy(OODModel):
    """
    Energy Score method for OOD detection.
    "Energy-based Out-of-distribution Detection"
    https://arxiv.org/abs/2010.03759

    This method assumes that the model has been trained with cross entropy loss :math:'CE(model(x))' where :math:'model(x)=(l_{c})_{c=1}^{C}' are the logits predicted for input :math: 'x'. 
    The implementation assumes that the logits are retreieved using the output with linear activation.

    The energy score for input :math:'x' is given by 
    .. math:: -\log \sum_{c=0}^C \exp(l_c)

    where 'model(x)=(l_{c})_{c=1}^{C}' are the logits predicted by the model on :math:'x'. 
    As always, training data is expected to have lower score than OOD data.  

    
    Args:
        batch_size: batch_size used to compute the features space
            projection of input data. 
            Defaults to 256.
    """
    def __init__(
        self, 
        output_activation: str = "linear", 
        batch_size: int = 256,
    ):
        super().__init__(output_activation=output_activation, 
                         batch_size=batch_size)

    def _score_tensor(
        self, 
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on 
        energy, namey :math:'-logsumexp(logits(inputs))'.

        Args:
            inputs: input samples to score

        Returns:
            scores
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"

        # compute logits (softmax(logits,axis=1) is the actual softmax output minimized using binary cross entropy)
        logits = self.feature_extractor(inputs)[0]
        scores = logsumexp(logits, axis=1)
        self.scores = -scores
        return self.scores

        