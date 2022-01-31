FROM tensorflow/tensorflow:latest
RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip python3-dev python3-opencv
RUN apt install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade wheel
RUN apt install git curl wget -y
# RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
COPY code/ /home/
COPY requirements.txt home/
COPY train.sh /home/
COPY yolo* /home/
COPY data.names /home/
RUN mkdir /home/data
WORKDIR /home
RUN python3 -m pip install -r requirements.txt
RUN chmod +x train.sh
EXPOSE 6006
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENTRYPOINT python /home/train.py
