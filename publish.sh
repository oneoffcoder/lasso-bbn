#!/bin/bash

SOURCE_DIST=/py-bbn/dist/pybbn-${LIB_VERSION}.tar.gz

buildCode() {
  echo "start the build"
  cd /pypi-lib \
    && make clean \
    && make \
    && python setup.py sdist bdist bdist_wheel \
    && twine check dist/* \
    && cd /pypi-lib/docs \
    && make html
}

updateVersion() {
  echo "replace version of software to ${LIB_VERSION}"
  sed -i "s/version='0.0.1'/version='${LIB_VERSION}'/g" /pypi-lib/setup.py
}

copyCredentials() {
  if [[ -f /pypi-lib/.pypirc ]]; then
    echo "copying over .pypirc"
    cp /pypi-lib/.pypirc /root/.pypirc
  fi
}

publish() {
  echo "python publish"

  if [[ -f /root/.pypirc ]]; then
    if [[ -f ${SOURCE_DIST} ]]; then
      echo "uploading source"
      cd /pypi-lib \
        && make clean \
        && python setup.py sdist \
        && twine upload --repository ${PYPI_REPO} ${SOURCE_DIST}
    else
      echo "no ${SOURCE_DIST} found!"
    fi
  else
    echo "no .pypirc found!"
  fi
}

cleanUp() {
  if [[ -f /root/.pypirc ]]; then
    echo "cleaning up"
    rm -f /root/.pypirc
  fi
}

build() {
  echo "python build"
  buildCode
  publish
}

conda init bash
. /root/.bashrc
updateVersion
copyCredentials
build
cleanUp

echo "done!"