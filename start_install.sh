pip install skia-python

git clone https://github.com/LordCocoro/jaxnerf.git

conda create --name jaxnerf python=3.7

echo "source activate jaxnerf" > ~/.bashrc

PATH /opt/conda/envs/env/bin:$PATH

conda install pip; pip install --upgrade pip setuptools wheel

pip install -r jaxnerf/requirements.txt

pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html