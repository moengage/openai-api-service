#!/bin/sh

uvicorn api.app:app --host 0.0.0.0 --workers 4
