
# DataHandler Tutorial

This tutorial demonstrates how to use the `DataHandler` class to load datasets in three different ways: using `torchvision`, `tensorflow-datasets`, and `huggingface`. We will use the CIFAR-10 dataset as an example.

First, you will need to load your `data_handler`, which can be done with the function `load_data_handler`. This function allows to load the correct object depending on the backend (`tensorflow` or `pytorch`).

When only one of these packages is installed in the environment, the function recognizes the backend:

```python
from oodeel.datasets import load_data_handler

data_handler = load_data_handler()
```
If both backends are installed in your environment, you have to specify the desired backend you will work with to `load_data_handler`:

```python
from oodeel.datasets import load_data_handler

data_handler = load_data_handler("torch") # or tensorflow
```

## 1. Loading CIFAR-10 with `torchvision`

The `TorchDataHandler` class provides an interface for loading datasets using `torchvision`. Below is an example of how to load CIFAR-10:

```python
# Load CIFAR-10 dataset
ds_train = data_handler.load_dataset(
    dataset_id="CIFAR10",
    hub="torchvision",
    load_kwargs={"root": data_path, "train": True, "download": True}
)

# Prepare the datasets
batch_size = 128
def preprocess_fn(examples):
    examples["input"] = examples["input"].float() / 255.0
    # Add normalization here if needed
    return examples

ds_train = data_handler.prepare(ds_train, batch_size, preprocess_fn, shuffle=True)
```

Note that `torchvision` is the default value of `hub`. We made it explicit in the example but you do not need to add `hub="torchvision"` in this case.

## 2. Loading CIFAR-10 with `tensorflow-datasets`

The `TFDataHandler` class provides an interface for loading datasets using `tensorflow-datasets`. Below is an example of how to load CIFAR-10:

```python
# Initialize the TFDataHandler
data_handler = load_data_handler("tensorflow")

# Load CIFAR-10 dataset
ds_train = data_handler.load_dataset(
    dataset_id="cifar10",
    hub="tensorflow-datasets",
    load_kwargs={"split": "train"}
)

# Prepare the datasets
batch_size = 128
def preprocess_fn(inputs):
    inputs["image"] = inputs["image"] / 255.0
    # Add normalization here if needed
    return inputs

ds_train = data_handler.prepare(ds_train, batch_size, preprocess_fn, shuffle=True)
```
This time we have to change our data_loader to make it compatible with tensorflow. Note that `tensorflow-datasets` is the default value of `hub`. We made it explicit in the example but you do not need to add `hub="tensorflow-datasets"` in this case.

## 3. Loading CIFAR-10 with `huggingface`

The `DataHandler` classes also support loading datasets from the `huggingface` hub. Depending on the backend, `DataHandler` will either return a `tf.data.Dataset` or a `datasets.Dataset`. Below is an example of how to load CIFAR-10.

### In **pytorch**:

```python
from oodeel.datasets import load_data_handler
from torchvision import transforms

# Initialize the TorchDataHandler
data_handler = load_data_handler("torch")

# Load CIFAR-10 dataset
ds_train = data_handler.load_dataset(
    dataset_id="cifar10",
    hub="huggingface",
    load_kwargs={"split": "train"}
)

# Prepare the datasets
batch_size = 128
def preprocess_fn(examples):
    # HF image datasets are not automatically transformed into tensors
    examples["img"] = transforms.PILToTensor()(examples["img"])
    examples["img"] = examples["img"].float() / 255.0
    # Add normalization here if needed
    return examples

ds_train = data_handler.prepare(ds_train, batch_size, preprocess_fn, shuffle=True)
```

### In **tensorflow**

The only difference for `tensorflow` is the `preprocess_fn`:
```python
import tensorflow as tf

def preprocess_cifar10(examples):
    examples["img"] = tf.cast(examples["img"], float) / 255.0
    return examples

# Exact same code as for pytorch here !
```

And the way we load `data_handler`:
```python
data_handler = load_data_handler("tensorflow") # instead of torch.
```
If only one of the two backends are installed, you do not even need to specify the backend ; `load_data_handler` will figure it out.

!!! Warning
    The conversion from `datasets.Dataset` to `tf.data.dataset` makes the first forward pass over the dataset quite slow. When using datasets from HugginFace in tensorflow, make sure to optimize your code so that you keep your `tf.data.dataset` in memory when it is used for several experiments.

This tutorial demonstrates how to load and preprocess datasets using the `DataHandler` class. You can adapt these examples to other datasets and frameworks as needed.

!!! info
    The `DataLoader` class also implements a bunch of features to apply operations to `torch.Dataset`s and `tf.data.dataset`s seamlessly with the same API. Check out the API references to learn more!
