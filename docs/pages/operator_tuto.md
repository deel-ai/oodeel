
Oodeel is designed to work with both Tensorflow and Pytorch models. However, we wanted to avoid duplicate code as much as possible.

Hence, we created the class `Operator` and the child classes `TFOperator` (API [here](/api/tf_operator)) and `TorchOperator` (API [here](/api/torch_operator)) to seamlessly perform basic operations on `tf.Tensor`or `torch.tensor`, for instance `mean`, `matmul`, `cat`, `softmax`...

!!! info
    Using this feature is not required to implement your own baselines with your favorite library, but only if you want your baseline to be usable with both Tensorflow and Pytorch.

The implementation shines when performing conditional import. Let's see how it works
## Example

### Basic usage

Suppose that you use either Tensorflow or PyTorch.

For torch, start with

```python
import torch

backend = "torch"
tensor = torch.ones((10,5))
```

or for Tensorflow

```python
import tensorflow as tf

backend = "tensorflow"
tensor = tf.ones((10,5))
```
Then you could conditionally load the correct `Operator`:

```python
if backend == "torch":
    from oodeel.utils import TorchOperator
    operator = TorchOperator()

elif backend == "tensorflow":
    from oodeel.utils import TFOperator
    operator = TFOperator()
```

!!! tip
    If you do not know in advance from which library the input tensor will come from, there is a nice function we implemented for you: `is_from` (see [here](/api/utils))

now it is possible to seamlessly use your `operator` to process your `tensor`:

```python
tensor_mean = operator.mean(tensor, dim=0)
```
!!! note
    We had to choose between Pytorch and Tensorflow syntax for `Operator` API. This object is to be mainly used by researchers wanting to make their baseline available for the community, so since Pytorch is the main library used for research, we adopted the Pytorch syntax.

### Get gradients

It is even possible to obtain the gradient using backprop. Let's take the previously instantiated `operator` or `tensor`. The following code is the same for Tensorflow or PyTorch.

```python
def power(x):
    return x ** 2

# Get the gradient
grads = operator.gradient(power, tensor)
```
The operator.gradient() function takes the function and the input tensor as arguments. If you want to perform library-specific operations, you should use the operator API. For instance:

```python
# Here we use tensorflow
tensor_1 = tf.ones((10,5))
tensor_2 = tf.ones((10,5))

# The following code does not depend on the underlying library
def mult(x):
    return operator.matmul(x, operator.transpose(tensor_2))

grads = operator.gradient(mult, tensor_1)

```

!!! note
    The idea is inspired by the great lib [EaerPy](https://github.com/jonasrauber/eagerpy), but we wanted to have closer control and make our own baked API to facilitate maintenance. In addition, we do not claim to reproduce the full API of tensorflow/pytorch/(soon jax ?) and implement the methods on the fly, if required by the baselines.

### Completing Operator API

As mentioned above, the API is far from exhaustive because we add methods on the fly, depending on the needs of OOD baselines. It is likely that you could need an unimplemented method on your way toward implementing a new baseline. **Don't panic:** there are two situations:

#### Contributor?

You are a contributor and you want your baseline to be officially part of Oodeel and available through Pypi. In that case, all you have to do is implement the method for both `TFOperator` and `TorchOperator` which is not *that* bad. For instance, take the method `one_hot`:

For `TFOperator`:

```python
def one_hot(tensor: Union[tf.Tensor, np.ndarray], num_classes: int) -> tf.Tensor:
    """One hot function"""
    return tf.one_hot(tensor, num_classes)
```

and for `TorchOperator`:

```python
def one_hot(
    tensor: Union[torch.Tensor, np.ndarray], num_classes: int
) -> torch.Tensor:
    """One hot function"""
    return torch.nn.functional.one_hot(tensor, num_classes)
```

#### Or simply User?

If you just want to use Oodeel for your research, you can simply use your favorite lib and develop your brand new baseline like usual, following the tuto [here](/pages/implementing_baselines_tuto) without bothering with `Operator`.
