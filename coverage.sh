#!/usr/bin/env bash
pytest **/tests.py --cov-report html --cov-config=.coveragerc --cov=. --disable-warning
