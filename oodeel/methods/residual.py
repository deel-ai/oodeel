import tensorflow as tf
from .base import OODModel
import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
from ..types import *


try:
    from kneed import KneeLocator
except ImportError:
    _has_kneed = False
    _kneed_not_found_err = ModuleNotFoundError("This function requires Kneed to be executed. Please run command `pip install kneed` or 'conda install -c conda-forge kneed'")
else:
    _has_kneed = True


class Residual(OODModel):
    """
        Compute the norm of the projection on residual dimensions for principal component analysis.
        Residual dimensions are the eigenvectors corresponding to the least eignevalues.
        Intuitively, this method assumes that feature representations of ID data occupy a low dimensional affine subspace :math:'P+c' of the feature space. 
        Specifically, the projection of ID data translated by :math:'-c' on the orthognoal complement :math:'P^\perp' is expected to have small norm.
        It allows to detect points whose feature representation lie far from the identified affine subspace, namely those points :math:'x' such that the projection on :math:'P^\perp' of :math:'x-c' has large norm.

        Args:
            output_layers_id: feature space on which to compute nearest neighbors. 
                Defaults to [-2].
            output_activation: output activation to use. 
                Defaults to None.
            res_dim: number of residual dimensions to consider. Let D be the dimension of the feature space and C the number of classes. If None and D-C>0, D-C is use. 
                Defaults to None.
            princ_dims: number of principal dimensions of in distribution features to consider. 
            If an int, must be less than the dimension of the feature space. 
            If a float, it must be in [0,1), it represents the ratio of explained variance to consider to determine the number of principal components. 
            If None, the kneedle algorithm is used to determine the number of dimensions.  
                Defaults to None.
            batch_size: batch_size used to compute the features space
                projection of input data. 
                Defaults to 256.
            
    """
    def __init__(
        self, 
        output_layers_id: List[int] = [-2], 
        output_activation: str = None,
        princ_dims: Union[int,float]= None,
        batch_size: int = 256,
    ):
        super().__init__(output_layers_id=output_layers_id,
                         output_activation=output_activation, 
                         flatten=True,
                         batch_size=batch_size)
        self.princ_dims=princ_dims
        
    def _fit_to_dataset(
        self, 
        fit_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
    ):
        """
        Compute principal components of feature representations and store the residual eigenvectors.


        Args:
            fit_dataset: input dataset (ID) to construct the index with.

        """
        features_train = self.feature_extractor(fit_dataset)[0]
        self.feature_dim=features_train.shape[1]
        self.center=tf.reduce_mean(features_train,axis=0)
        #estimate covariance matrix of features
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features_train-self.center)
        # compute eigenvalues and eigenvectors of empirical covariance matrix
        eig_vals, eigen_vectors = eigh(ec.covariance_)
        # allow to use Kneedle to find res_dim
        self.eigenvalues=eig_vals

        # TODO maybe it would be interesting to allow to pass a ratio of explained variance
        # given exp_var in [0,1], then res_dim would be the least index D such that 
        # sum(eig_vals[D:])/sum(eig_vals)>exp_var
        if self.princ_dims is None:
            if not _has_kneed:
                raise _kneed_not_found_err
            self.kneedle = KneeLocator(range(len(eig_vals)), eig_vals, S=1.0, curve="convex", direction="increasing")
            self.res_dim=self.kneedle.elbow
            assert 0<self.res_dim and self.res_dim<self.feature_dim, f"Found invalid number of residual dimensions ({self.res_dim}) "
            print(f"Found an elbow point for {self.feature_dim-self.res_dim} principal dimensions inside the {self.feature_dim} dimensional feature space.")
            print("It is assumed that spectrum is convex to find this number with the kneedle algorithm, please verify!")
            print("You can visualize this elbow by calling the method '.plot_spectrum()' of this class")
        elif isinstance(self.princ_dims, int):
            assert self.princ_dims<self.feature_dim, f"if 'princ_dims'(={self.princ_dims}) is an int, it must be less than feature space dimension ({self.feature_dim})"
        elif isinstance(self.princ_dims, float):
            assert 0<=self.princ_dims and self.princ_dims<1,  f"if 'princ_dims'(={self.princ_dims}) is a float, it must be in [0,1)"   
            explained_variance=np.cumsum(np.flip(eig_vals)/np.sum(eig_vals))
            self.princ_dims=np.where(explained_variance>self.princ_dims)[0][0]
            self.res_dim=self.feature_dim-self.princ_dims
 

        self.res = tf.constant(np.ascontiguousarray(eigen_vectors[:,:self.res_dim],np.float32))


    def plot_spectrum(self)-> None:
        if hasattr(self, "kneedle" ):
            self.kneedle.plot_knee()
            plt.title(f"Found elbow at dimension {self.kneedle.elbow}\n {self.feature_dim-self.kneedle.elbow} principal dimensions")
        else:
            plt.plot(self.eigenvalues)
            plt.axvline(x=self.res_dim,color="r", linestyle="--", label=f"Number of principal dimensions = {self.princ_dims} ")
            plt.legend()
            plt.title(f"Sorted eigen values")



    def _score_tensor(
        self, 
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on 
        on the residual norm (PCA) in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            scores
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"
        # compute predicted features
        features = self.feature_extractor(inputs)[0]
        # compute norm of residual component
        # compute coordinates of projections wrt to scaled eigenvectors v_j
        # if features is NxD and self.res is DxR with R=res_dim, 
        # then input_res is NxR and input_res[i,j] is the dot product of features[i]  with eigenvector v_j
        #TODO Tensor Compatibility: use of TF tensors to accelerate matrix multiplication!
        res_coordinates=tf.matmul(features-self.center,self.res)
        # taking the norm of the coordinates, which amounts to the norm of the projection since the eigenvectors form an orthornomal basis
        res_norm=tf.norm(res_coordinates, axis=1)
        scores= np.array(res_norm)
        return scores

        
