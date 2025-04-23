#!/bin/sh

PARAMETER_NAME="BEDROCK_CERTIFICATE"
parameter_value=$(aws ssm get-parameter --name $PARAMETER_NAME --with-decryption --query Parameter.Value --output text)

cafile=/usr/local/lib/python3.12/site-packages/certifi/cacert.pem

if [ -z "$parameter_value" ]; then
  echo "Failed to retrieve parameter value from AWS Parameter Store"
  exit 1
fi

echo "$parameter_value" > $cafile

echo "Content retrieved from AWS Parameter Store and written to file successfully"

uvicorn api.app:app --host 0.0.0.0
