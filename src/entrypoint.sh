#!/bin/sh


cafile=/usr/local/lib/python3.12/site-packages/certifi/cacert.pem

echo $BEDROCK_CERTIFICATE >> $cafile

uvicorn api.app:app --host 0.0.0.0
