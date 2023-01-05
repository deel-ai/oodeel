# Index

Mainly you could copy the README.md here. However, you should be careful with:

- The banner section is different
- Link to assets (handling dark mode is different between GitHub and the documentation)
- Relative links

<!-- Banner section -->
<div align="center">
    <img src="./assets/banner_dark.png#only-dark" width="75%" alt="lib banner" align="center" />
    <img src="./assets/banner_light.png#only-light" width="75%" alt="lib banner" align="center" />
</div>
<br>

<!-- Badge section -->
<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.6, 3.7, 3.8-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

<!-- Short description of your library -->
<p align="center">
  <b>Libname</b> is a Python toolkit dedicated to make people happy and fun.

  <!-- Link to the documentation -->
  <br>
  <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"><strong>Explore Libname docs »</strong></a>
  <br>

</p>

## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [👨‍🎓 Creator](#-creator)
- [🗞️ Citation](#-citation)
- [📝 License](#-license)

## 🔥 Tutorials

We propose some tutorials to get familiar with the library and its api:

- [Getting started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/<libname>/blob/master/docs/notebooks/demo_fake.ipynb) </sub>

You do not necessarily need to register the notebooks on the GitHub. Notebooks can be hosted on a specific [drive](https://drive.google.com/drive/folders/1DOI1CsL-m9jGjkWM1hyDZ1vKmSU1t-be).

## 🚀 Quick Start

Libname requires some stuff and several libraries including Numpy. Installation can be done using Pypi:

```python
pip install libname
```

Now that Libname is installed, here are some basic examples of what you can do with the available modules.

### Print Hello World

Let's start with a simple example:

```python
from libname.fake import hello_world

hello_world()
```

### Make addition

In order to add `a` to `b` you can use:

```python
from libname.fake import addition

a = 1
b = 2
c = addition(a, b)
```

## 📦 What's Included

A list or table of methods available

## 👍 Contributing

Feel free to propose your ideas or come and contribute with us on the Libname toolbox! We have a specific document where we describe in a simple way how to make your first pull request: [just here](CONTRIBUTING.md).

## 👀 See Also

This library is one approach of many...

Other tools to explain your model include:

- [Random](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library exclusively dedicated to explaining neural networks.
- [deel-lip](https://github.com/deel-ai/deel-lip) a Python library for training k-Lipschitz neural networks on TF.
- [Influenciae](https://github.com/deel-ai/influenciae) Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
- [deel-torchlip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

<img align="right" src="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10#only-dark" width="25%" alt="DEEL Logo" />
<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png#only-light" width="25%" alt="DEEL Logo" />
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 👨‍🎓 Creators

If you want to highlights the main contributors


## 🗞️ Citation

If you use Libname as part of your workflow in a scientific publication, please consider citing the 🗞️ [our paper](https://www.youtube.com/watch?v=dQw4w9WgXcQ):

```
@article{rickroll,
  title={Rickrolling},
  author={Some Internet Trolls},
  journal={Best Memes},
  year={ND}
}
```

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
