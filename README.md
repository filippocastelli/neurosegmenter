# neuroseg

### creating an Anaconda env
Can use the provided yml env configurator to build a ready-to-use anaconda environment.

```
conda env create --file conda_env.yml

conda activate neuroseg
```

### Building a docker container

first build the package
```
python setup.py bdist_wheel
```
then build the docker image
```
docker build -t neuroseg:latest .
```

### Running an example script
```
sudo docker run -it -v /home/castelli/neuroseg/examples:/opt/examples python /opt/examples/example_script.py
```

### resolving TF 2.7.0 issues: wrong CUDA libs

If TF 2.7 doesn't find the correct CUDA libs in an anaconda env it might depend on how libraries are loaded by TF after 2.7

just set ```LD_LIBRARY_PATH``` 

```
export LD_LIBRARY_PATH=/home/phil/anaconda3/envs/neuroseg/lib
```
