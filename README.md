# OODEEL

## Current features:

* 2 classes, OODModel and OODModelWID 
* methods:
    *   Post-hoc without ID data: Maximum Logit Score
    *   Post-hoc with ID data: Deep KNN
* Feature extractor of an arbitrary number of internal activations
* training script for a convnet on mnist
* Demo notebook on MNIST vs Fashion MNIST
* Experiments object for one dataset ID vs one dataset OOD, with AUROC as metric (ID MNIST, OOD fashion MNIST)
* Experiments object for leave $k$-classes type benchmark. Implemented for MNIST. Can arbitrary select classes as ID, the others are considered OOD.

The features are tested on the notebooks. 

## Todos

### On the core library

* Think thoroughly about the structure of abstract classes and the current functions. Think about the workflow. Is it ok like this?
* How / should we include post hoc methods that use extra data? (e.g. outlier exposure that uses some OOD data.)
* Include unit tests
* Add type on function signatures
* Tensorflow ? Pytorch ? (Jax ?) Use Eagerpy ?(seems difficult since methods are applied on models, and datasets will probably be managed by a data loader from tf or pytorch.) 
* Check for memory optimization: currently some keras model are dynamically created, which is prone to memory leaks, I think.
* Use tf dataset or torchvision for managing datasets
* Manage checkpointing for Single dataset expe, to avoid having to retrain a model from scratch when possible.

### Further development

* Add new baselines
* Add new metrics
* Add new datasets
* Add new training scripts (for other backbones like Resnet)
* Weights storage system? 
* Automatic threshold selection? (exploratory)

