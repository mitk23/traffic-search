FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

USER root

RUN apt-get update && apt-get -y upgrade
RUN DEBIAN_FRONTEND=noninteractive apt install -y curl git python3.9 python3-pip
RUN python3.9 -m pip install -U wheel setuptools

COPY ./requirements.txt /tmp
RUN python3.9 -m pip install -r /tmp/requirements.txt
