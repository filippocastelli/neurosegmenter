FROM tensorflow/tensorflow:2.3.2-gpu

COPY . /opt/neuroseg/
WORKDIR /opt/neuroseg

RUN echo "$(tail -n +2 requirements.txt)" > requirements.txt
RUN pip install -r requirements.txt
RUN pip install wheel


RUN python setup.py bdist_wheel
RUN pip install dist/*

CMD ["python"]