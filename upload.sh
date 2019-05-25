#!/usr/bin/env bash

rm -rf .eggs build dist ikfs_anomaly_detector.egg-info

python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
