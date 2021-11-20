arrIN=(${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS//:/})
arrIN=(${arrIN//grpc\/\//})
arrIN=(${arrIN//8470/})
TPU_IP=http://${arrIN[0]}:8475/requestversion/tpu_driver_nightly

curl -X POST "${TPU_IP}" || echo ":C"
