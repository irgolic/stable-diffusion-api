#!/bin/bash
poetry install --with=test
poetry run pytest -m e2e --e2e