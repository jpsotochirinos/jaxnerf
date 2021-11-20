# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import requests
from jax import config

path = os.getenv('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS')
path = path.split(':')
url = 'http://'+path[1][2:]+':8475/requestversion/tpu_driver_nightly'
reqq = requests.post(url)
print(reqq)
TPU_DRIVER_MODE = 1
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = os.getenv('KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS')
