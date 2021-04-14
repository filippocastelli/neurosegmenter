FROM tensorflow/tensorflow:2.3.2-gpu
RUN pip install wheel

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && rm /requirements.txt && rm -rf /root/.cache/pip/
COPY dist/*.whl /opt/neuroseg/dist/

#RUN echo "$(tail -n +2 requirements.txt)" > requirements.txt
#RUN pip install -r requirements.txt

#RUN python setup.py bdist_wheel

RUN pip install /opt/neuroseg/dist/*.whl && rm -rf /root/.cache/pip/

CMD ["python"]
