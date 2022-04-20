# scikit-ika

[![PyPI](https://img.shields.io/pypi/v/scikit-ika.svg)](https://pypi.python.org/pypi/scikit-ika) [![Build Status](https://travis-ci.com/scikit-ika/scikit-ika.svg?branch=master)](https://travis-ci.com/scikit-ika/scikit-ika)

A real-time adaptive predictive system for evolving data streams. [Learn More](https://scikit-ika.github.io/about/)

### Requirements

Make sure the following dependencies are installed:

* Python &ge; 3.6

* C++ toolchain supporting C++14 (g++ 7+, CMake)

### Installing

From PyPI

```bash
pip install scikit-ika
```

From source

```bash
git clone https://github.com/scikit-ika/scikit-ika.git --recursive
pip install ./scikit-ika
```

### Building on Windows x64

1. Install the Visual Studio 2022 Community Edition (or later). Make sure you tick the box for "Desktop C++" development.
2. Install the latest CMake.
3. Install the latest 64bit Python. Make sure you tick the box to add Python to the system path.
4. Install Git.
5. Run "Developer PowerShell for VS 2022" (or whatever your Visual Studio is).
6. Checkout the code: `git clone https://github.com/scikit-ika/scikit-ika.git --recursive`
7. `cd scikit-ika`
8. `pip install .`

### Note

`scikit-ika` is still in its infancy. The APIs are unstable and the documentation is scarse.

Guides on usages and parameter tunning are on its way. Meanwhile, please feel free to
raise issues or email us if you have any questions.

### Acknowledgement

This project is inspired by and has dependencies on the following projects:

* [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow)
* [streamDM in C++](https://github.com/huawei-noah/streamDM-Cpp)
