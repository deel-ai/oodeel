# OODEEL

## Current features:

* 1 class OODModel 
* 2 methods (child classes of OODModel):
    *   Post-hoc without ID data: Maximum Logit Score
    *   Post-hoc with ID data: Deep KNN
* class Feature extractor of an arbitrary number of internal activations 
* dataset handler to merge ID and OOD datasets, or two filter a dataset based on its labels
* training script for a convnet on mnist
* Demo notebook on MNIST vs Fashion MNIST
* Experiments example for one dataset ID vs one dataset OOD, with AUROC as metric (ID MNIST, OOD fashion MNIST)
* Experiments example for leave $k$-classes type benchmark. Implemented for MNIST. Can arbitrary select classes as ID, the others are considered OOD.

The features are tested on the notebooks. 

## Todos

### On the core library

* How / should we include post hoc methods that use extra data? (e.g. outlier exposure that uses some OOD data.)
* Include unit tests
* Tensorflow ? Pytorch ? Both ? Idea: model conversion with ONNX to be able to handle both types of models while only using one of the two lib for dev.
* Use tf dataset or torchvision for managing datasets
* Manage checkpointing for Single dataset expe, to avoid having to retrain a model from scratch when possible.

### Further development

* Viz module
* Add new baselines
* Add new metrics
* Add new training scripts (for other backbones like Resnet)
* Weights storage system? 
* Automatic threshold selection? 

