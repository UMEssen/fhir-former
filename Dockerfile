FROM nvcr.io/nvidia/tensorflow:23.06-tf2-py3
MAINTAINER merlin.engelke@uk-essen.de

RUN apt update && \
    apt upgrade -y

COPY app/requirements.txt .
RUN python -m pip install -r requirements.txt
RUN mkdir /autopilot
RUN chmod a+rwx -R /autopilot