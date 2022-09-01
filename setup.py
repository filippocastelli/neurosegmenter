from setuptools import setup, find_packages
import importlib
import packaging
import os
from importlib.metadata import version, PackageNotFoundError

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

with open("neuroseg/version", "r") as fh:
    __version__ = fh.read()

# if tensorflow already installed, via conda don't install tensorflow-gpu
is_conda = 'CONDA_PREFIX' in os.environ or 'CONDA_DEFAULT_ENV' in os.environ

tf_line = [line for line in requirements if line.startswith("tensorflow-gpu==")][0]
tfvers = tf_line.split("==")[1]

tf_installed = importlib.util.find_spec("tensorflow") is not None
tf_gpu_installed = importlib.util.find_spec("tensorflow-gpu") is not None

if tf_gpu_installed:
    if version("tensorflow-gpu") == tfvers:
        # do nothing
        pass

if tf_installed:
    print("tensorflow already installed")
    if version("tensorflow") == tfvers and is_conda:
        # no need to install tensorflow-gpu if conda tensorflow is installed
        requirements = [line for line in requirements if line != tf_line]

setup(
    name="neuroseg",
    version=__version__,
    description="NeuroSeg",
    packages=find_packages(exclude=("tests",)),
    package_data={"": ["version", "utils/animals.txt", "utils/adjectives.txt"]},
    # py_modules = ["omereader", "omewriter", "omexml"],
    # package_dir = {"": "pyometiff"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        # "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        # "Operating System :: Os Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    test_requirements = ["pytest", "mock", "pudb"],
    url="https://github.com/filippocastelli/neuroseg",
    author="Filippo Maria Castelli",
    author_email="castelli@lens.unifi.it",
    entry_points={
        "console_scripts": [
            "neuroseg_train = neuroseg.train.__main__:main",
            "neuroseg_predict = neuroseg.predict.__main__:main"
        ]
    }
)
