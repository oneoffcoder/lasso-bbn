#!/bin/bash

DOCKER_FILE=Dockerfile
DOCKER_REPO=lassobbn
DOCKER_TAG=local
ALIB_VERSION=version
APYPI_REPO=repo

while getopts v:r: option
do
  case "${option}"
  in
  v) ALIB_VERSION=${OPTARG};;
  r) APYPI_REPO=${OPTARG};;
esac
done

if [[ "version" == ALIB_VERSION || "repo" == $APYPI_REPO ]]; then
  echo "Usage: ./build.sh -r [pypi|testpypi] -v [version]"
  echo "     -r repository, pypi or testpypi"
  echo "     -v version e.g. 0.2.5"
else
  docker build --no-cache \
    -f $DOCKER_FILE \
    --build-arg ALIB_VERSION=$ALIB_VERSION \
    --build-arg APYPI_REPO=$APYPI_REPO \
    -t ${DOCKER_REPO}:${DOCKER_TAG} .
fi