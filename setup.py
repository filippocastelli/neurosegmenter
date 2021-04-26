from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()
    
with open("neuroseg/version", "r") as fh:
    __version__ = fh.read()
    
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
      # install_requires=["numpy",
      #                   "pandas",
      #                   "tensorflow>2.1",
      #                   "tensorflow-addons",
      #                   "scikit-image",
      #                   "scikit-learn",
      #                   "matplotlib",
      #                   "tifffile>2020.10.1",
      #                   "tqdm",
      #                   "batchgenerators"],
      # extras_require={
      #     "dev": ["pytest>3.7","mock"],
      #     },
      
      url="https://github.com/filippocastelli/neuroseg",
      author="Filippo Maria Castelli",
      author_email="castelli@lens.unifi.it",
      )
