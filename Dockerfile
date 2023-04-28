FROM ubuntu:20.04
MAINTAINER merlin.engelke@uk-essen.de

RUN apt update && \
    apt upgrade -y && \
    apt install -y python3.9 && \
    apt install -y python3-pip

COPY app/requirements.txt .
RUN python3.9 -m pip install -r requirements.txt