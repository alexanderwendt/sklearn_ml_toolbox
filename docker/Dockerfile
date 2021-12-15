FROM python:3.8

RUN apt-get -y update
RUN apt-get -y install curl
RUN apt-get -y install python3-pip
RUN pip install virtualenv
RUN apt -y install git

run mkdir -p /sklearn-ml-toolbox
COPY ./bootstrap.sh /sklearn-ml-toolbox
WORKDIR /sklearn-ml-toolbox
RUN chmod +x *.sh
RUN ./bootstrap.sh

#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility