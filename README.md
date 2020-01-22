# Introduction
PyOpt is a Python optimization framework written as a Python package.

# Index

- [About](#about)
- [Usage](#usage)
  - [Installation](#installation)
    - [Python Package Dependencies](#python-package-dependencies)
- [Development](#development)
  - [Validation Tests](#validation-tests)
- [Community](#community)
  - [Contribution](#contribution)
- [Credit/Acknowledgment](#creditacknowledgment)
- [License](#license)

# About
PyOpt is a python package meant to make the boilerplate for optimization of numerical models as minimal as possible. The idea is that PyOpt supplies you with the support structure for inspecting optimization jobs no matter the optimization routine you use.

# Usage
This software is meant to be used as a Python package. To begin working with PyOpt, simply clone the PyOpt repository and import the PyOpt package as normal.

### Installation

PyOpt has the following dependencies:
+ `NumPy`, version = 1.16.4
+ `Matplotlib`, version = 3.1.1
+ `Py-OpenCV`, version = 3.4.2
+ `ImageIO`, version = 2.5.0

If you wish to work on PyOpt, you will also need:
+ PyTest, version = 5.3.2

Additionally, we use the following libraries heavily with PyOpt for the following optimization methods:
+ `PySwarms` - Particle swarm optimization. Version = 1.1.0
+ `scipy.optimize.minimize` - Nelder-Mead (Simplex) method. `Scipy` version = 1.2.1

# Development
Thank you for considering contributing to the PyOpt project! Please contact Dylan Colli at dylanfrankcolli@gmail.com.

PyOpt uses the [PyTest](https://docs.pytest.org/en/latest/index.html) framework for validation of functionality. PyTest is not part of Python's standard library and thus needs to be installed before validation of changes can be done. To do so, visit [this page](https://docs.pytest.org/en/latest/getting-started.html) to install PyTest.

### Validation Tests
It is important to validate any code before it is committed (and develop new validation tests as appropriate!). To run a full validation test for the project, execute the following from the project's root directory:
```
$ pytest
```

If you would like to view the output of the tests, simply add `-s` to the previous command.

# Community

Hello! Thanks for taking the time to read through the documentation to learn more about the PyOpt project. We welcome any sort of dialogue about the project and if you have any questions or concerns, feel free to email Dylan Colli at dylanfrankcolli@gmail.com or see below for issue tracking and feature requests.

### Contribution

Your contributions are always welcome and appreciated. Following are the things you can do to contribute to this project:

1. **Report a bug**
If you think you have encountered a bug, feel free to report it using the "Issues" tab in the bitbucket repository and I will take care of it.

2. **Request a feature**
You can also request the addition of a feature using the issue tracker by selecting "proposal" when prompted by the "Kind" dialogue.

# Credit/Acknowledgment
The following is a list of contributors to the PyOpt project

Dylan Colli - dfco222@g.uky.edu<br/>

# License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. The licensce is included in the `COPYING.txt` document.