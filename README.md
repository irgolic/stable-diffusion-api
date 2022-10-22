# Stable Diffusion Server

[![OpenApi](https://img.shields.io/badge/OpenApi-3.0.2-orange)](https://editor.swagger.io/?url=https://raw.githubusercontent.com/irgolic/stable-diffusion-server/master/openapi.yml)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irgolic/stable-diffusion-server/blob/master/colab_runner.ipynb)

Simple backend to serve Txt2Img, Img2Img and Inpainting with any model published on [Hugging Face](https://huggingface.co/models).
The aim is to provide a lightweight, easily extensible backend for image generation.

## Features

Supported parameters:
- `prompt`: text prompt **(required)**, e.g. `corgi with a top hat`
- `negative_prompt`: negative prompt, e.g. `monocle`
- `model_id`: model name, default `CompVis/stable-diffusion-v1-4`
- `model_provider`: model provider, currently only `huggingface` is supported
- `steps`: number of steps, default `20`
- `guidance`: relatedness to `prompt`, default `7.5`
- `scheduler`: either `plms`, `ddim`, or `k-lms`
- `seed`: randomness seed for reproducibility, default `None`
- `safety_filter`: enable safety checker, default `true`

Txt2Img also supports:
- `width`: image width, default `512`
- `height`: image height, default `512`

Img2Img also supports:
- `initial_image`: `blob_id` of image to be transformed **(required)**
- `strength`: how much to change the image, default `0.8`

Inpainting also supports:
- `initial_image`: `blob_id` of image to be transformed **(required)**
- `mask`: `blob_id` of mask image **(required)**

Img2Img can reference a previously generated image's `blob_id`. 
Alternatively, `POST /blob` to upload a new image, and get a `blob_id`.

## Usage

Generate any client library from the [OpenApi](
https://editor.swagger.io/?url=https://raw.githubusercontent.com/irgolic/stable-diffusion-server/master/openapi.yml) specification.

### Authentication

The token is passed either among query parameters (`/txt2img?token=...`), or via the `Authorization` header 
as a `Bearer` token [(OAuth2 Bearer Authentication)](https://swagger.io/docs/specification/authentication/bearer-authentication/).

To disable authentication and allow generation of public tokens at `POST /token/all`,
set environment variable `ENABLE_PUBLIC_ACCESS=1`.

To allow users to sign up at `POST /user`, 
set environment variable `ENABLE_SIGNUP=1`. 
Registered users can generate their own tokens at `POST /token/{username}`.

### Synchronous Interface

For convenience, the server provides synchronous endpoints at `GET /txt2img`, `GET /img2img`, and `GET /inpaint`.

To print a browser-accessible URL upon startup (i.e., `http://localhost:8000/txt2img?prompt=corgi&steps=5?token=...`), 
set environment variable `PRINT_LINK_WITH_TOKEN=1` (set by default in `.env.example`).

The server will wait for the model to download and generate an image before returning the request.
It is preferable to use the asynchronous endpoints for production use.

### Asynchronous Interface (recommended)

#### Invocation

`POST /task` with either `Txt2ImgParams`, `Img2ImgParams` or `InpaintParams` to start a task, and get a `task_id`. 
The model will be downloaded and cached, and the task will be queued for execution.

#### Job Status

`GET /task/{task_id}` to get the last `event` broadcast by the task, or subscribe to the websocket endpoint `/events?token=<token>` to get a stream of events as they occur.

Event types:
- PendingEvent
- StartedEvent
- FinishedEvent (with `blob_id`)
- CancelledEvent (with `reason`)

#### Results

A FinishedEvent contains an `image` field including its `blob_id` and `parameters_used`.

`GET /blob/{blob_id}` to get the image in PNG format.

## Roadmap

Help wanted.

- [x] CUDA and CPU support
- [x] Seed parameter
- [ ] Inpainting
- [ ] Cancel task endpoint
- [ ] GFPGAN postprocessing (fix faces)
- [ ] Speed up image generation
- [ ] Progress update events (model download, image generation every n steps)
- [ ] Custom tokenizers, supporting `(emphasis:1.3)` and `[alternating,prompts,0.4]`
- [ ] More model providers
- [ ] Other GPU support – testers needed!

## Installation

Install a virtual environment with python 3.10 and poetry.

### Python Virtual Environment

Install python 3.10 with your preferred environment creator.

#### Conda (for example)

Install [MiniConda](https://docs.conda.io/en/latest/miniconda.html) and create a new environment with python 3.10.

```bash
conda create -n sds python=3.10
conda activate sds
```

### Poetry

Install [Poetry](https://python-poetry.org/docs/#installation) and install the dependencies.

```bash
poetry install
```

## Running

### Environment Variables

See `.env.example` for a list of example environment variable values.
Copy `.env.example` to `.env` and edit as needed (make sure to regenerate the secrets).

- `SECRET_KEY`: The secret key used to sign the JWT tokens.
- `PRINT_LINK_WITH_TOKEN`: Whether to print a link with the token to the console on startup.
- `ENABLE_PUBLIC_ACCESS`: Whether to enable public token generation (anything except empty string enables it).
- `ENABLE_SIGNUP`: Whether to enable user signup (anything except empty string enables it).
- `REDIS_PORT`: The port of the Redis server.
- `REDIS_HOST`: The host of the Redis server.
- `REDIS_PASSWORD`: The password of the Redis server.
- `HUGGINGFACE_TOKEN`: The token used by the worker to access the Hugging Face API.

### Docker Compose

Run the API and five workers, with redis as intermediary, and docker-compose to manage the containers.

```bash
make run
```

### Multi Process

Or invoke processes on multiple machines, starting the API with:

```bash
poetry run uvicorn stable_diffusion_server.api.redis_app:app
```

And the worker(s) with:

```bash
poetry run python -m stable_diffusion_server.engine.worker.redis_worker
```

### Single Process

Run the API and worker in a single process. API requests will block until the worker is finished.

```bash
poetry run uvicorn stable_diffusion_server.api.in_memory_app:app
```