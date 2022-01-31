FROM ubuntu:latest
RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip python3-dev
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade wheel
RUN pip3 install tensorboard
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENTRYPOINT tensorboad --logdir=/data/logs --port=6006 --bind_all
