FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-7

RUN apt-get update -y

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add 

RUN apt-get update -y

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y

RUN apt install libgl1-mesa-glx -y

RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN apt-get install -y wget git python3.7 python-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install skia-python

RUN git clone https://github.com/LordCocoro/jaxnerf.git

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r jaxnerf/requirements.txt

RUN pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

ENV MODELS_BUCKET='gs://nerf-bucket/models'

ENV CHECKPOINT_BUCKET='gs://nerf-bucket/chekpoint'

RUN cd jaxnerf

WORKDIR /jaxnerf

RUN git fetch

RUN git pull

RUN cd ..

WORKDIR /

EXPOSE 3000

CMD [ "python","-m", "jaxnerf.app" ]