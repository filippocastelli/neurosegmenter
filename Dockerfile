FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
    
RUN apt-get -y update
RUN apt-get -y install build-essential
# needed for pyvista
RUN apt-get -y install libxrender-dev libgl1 && apt-get -y update
RUN apt-get -y install python3.8 python3.8-distutils python3.8-dev python3.8-venv curl && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1  && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
# RUN apt-get -y install ocl-icd-libopencl1
RUN python -m venv neuroseg_env && . neuroseg_env/bin/activate
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && rm /requirements.txt && rm -rf /root/.cache/pip/
COPY dist/*.whl /opt/neuroseg/dist/
RUN pip install /opt/neuroseg/dist/*whl && rm -rf /root/.cache/pip

CMD ["python"]
