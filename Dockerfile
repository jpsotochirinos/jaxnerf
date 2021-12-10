FROM tensorflow/tensorflow:2.7.0

RUN apt-get update -y

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y

RUN apt-get install wget git python3.7 'ffmpeg' 'libsm6' 'libxext6' libgl1-mesa-glx  -y

RUN pip install --upgrade pip setuptools wheel

RUN pip install skia-python

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN wget https://sochperu.com/key/spe3d-331118-043d52d11808.json
RUN gcloud auth activate-service-account --key-file spe3d-331118-043d52d11808.json
RUN rm -rf spe3d-331118-043d52d11808.json

RUN git clone https://github.com/LordCocoro/jaxnerf.git

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r jaxnerf/requirements.txt

RUN pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

ENV CLOUD_TPU_TASK_ID="0"
ENV TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
ENV TPU_HOST_BOUNDS="1,1,1"
ENV TPU_MESH_CONTROLLER_PORT="8476"
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices"


RUN cd ..

WORKDIR /

EXPOSE 3000

CMD [ "python","-m", "jaxnerf.app" ]