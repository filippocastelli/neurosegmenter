FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y \
    apt-utils \
    unzip \
    tar \
    curl \
    xz-utils \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    ;

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get -y update
RUN apt-get -y install build-essential
RUN apt-get -y install python3.8 python3.8-distutils python3.8-dev python3.8-venv curl && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1  && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
RUN python -m venv neuroseg_env && . neuroseg_env/bin/activate
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && rm /requirements.txt && rm -rf /root/.cache/pip/
COPY dist/*.whl /opt/neuroseg/dist/
RUN pip install /opt/neuroseg/dist/*whl && rm -rf /root/.cache/pip
CMD ["python"]