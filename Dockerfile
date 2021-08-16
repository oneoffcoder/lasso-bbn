FROM oneoffcoder/python-java:latest

LABEL author="Jee Vang, Ph.D."
LABEL email="vangjee@gmail.com"

ARG ALIB_VERSION
ARG APYPI_REPO

ENV LIB_VERSION=$ALIB_VERSION
ENV PYPI_REPO=$APYPI_REPO

RUN apt-get update \
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install graphviz libgraphviz-dev pkg-config -y
COPY . /pypi-lib
RUN pip install -r /pypi-lib/requirements.txt
RUN /pypi-lib/publish.sh
