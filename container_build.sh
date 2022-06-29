#!/bin/bash
CONDA_ENV=neuroseg_pip
CONDA_PATH=/home/castelli/anaconda3
DOCKER_REGISTRY=atlante.lens.unifi.it:5000
DOCKERFILE=Dockerfile
DOCKERFILE_OPENCL=Dockerfile.opencl


sudo rm -r dist build;
source ${CONDA_PATH}/bin/activate ${CONDA_PATH}/envs/neuroseg_pip
echo using python
which python;
vers=`cat neuroseg/version`;
container_name=neuroseg:${vers}
container_name_opencl=${container_name}"-opencl"
container_name_reg=${DOCKER_REGISTRY}/${container_name}
container_name_reg_opencl=${DOCKER_REGISTRY}/${container_name_opencl}

python setup.py bdist_wheel;

echo Building container ${container_name_reg};
sudo nvidia-docker build -t ${container_name_reg} -f ${DOCKERFILE} .;

echo Building container ${container_name_reg_opencl};
sudo nvidia-docker build -t ${container_name_reg_opencl} -f ${DOCKERFILE_OPENCL} .;

cat enabling tunnel...;
sshuttle -r castelli@atlante.lens.unifi.it -x liquid.lens.unifi.it 150.217.0.0/16 150.217.157.89 -D;
sudo nvidia-docker push ${container_name_reg};
sudo nvidia-docker push ${container_name_reg_opencl};
