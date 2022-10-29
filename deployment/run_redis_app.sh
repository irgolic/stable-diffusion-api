#!/bin/bash
poetry run uvicorn stable_diffusion_api.api.redis_app:app --host 0.0.0.0