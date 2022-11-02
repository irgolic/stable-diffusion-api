# 👸 Stable Diffusion API 🐕

[![OpenApi](https://img.shields.io/badge/OpenApi-3.0.2-orange)](https://editor.swagger.io/?url=https://raw.githubusercontent.com/irgolic/stable-diffusion-api/master/openapi.yml)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irgolic/stable-diffusion-api/blob/master/colab_runner.ipynb)
[![Discord](https://discordapp.com/api/guilds/1027703335224098857/widget.png?style=shield)](https://discord.gg/UXQfCRpYSC)

Lightweight API for txt2Img, img2Img and inpainting, built with [🤗 diffusers](https://github.com/huggingface/diffusers/).

## Quickstart

Run it  on [Google Colab](https://colab.research.google.com/github/irgolic/stable-diffusion-api/blob/master/colab_runner.ipynb), 
or see [running instructions](#running) to use it locally.

The API prints a link on startup, which invokes txt2img when visited in the browser. 

Join our [Discord server](https://discord.gg/UXQfCRpYSC) for help, or to let me know what you'd like to see.

## Usage

Visit the API url in your browser [synchronously](#synchronous-interface) at 
`/txt2img`, `/img2img` or `/inpaint`, and append parameters with `?prompt=corgi&model=...`.

Or generate a client library in any popular programming language with the [OpenApi specification](
https://editor.swagger.io/?url=https://raw.githubusercontent.com/irgolic/stable-diffusion-api/master/openapi.yml), 
and implement it [asynchronously](#asynchronous-interface).

### Features

Supported parameters:
- `prompt`: text prompt **(required)**, e.g. `corgi with a top hat`
- `negative_prompt`: negative prompt, e.g. `monocle`
- `model`: model name, default `CompVis/stable-diffusion-v1-4`
- `steps`: number of steps, default `20`
- `guidance`: relatedness to `prompt`, default `7.5`
- `scheduler`: either `plms`, `ddim`, or `k-lms`
- `seed`: randomness seed for reproducibility, default `None`
- `safety_filter`: enable safety checker, default `true`

Txt2Img also supports:
- `width`: image width, default `512`
- `height`: image height, default `512`

Img2Img also supports:
- `initial_image`: URL of image to be transformed **(required)**
- `strength`: how much to change the image, default `0.8`

Inpainting also supports:
- `initial_image`: URL of image to be transformed **(required)**
- `mask`: URL of mask image **(required)**

`POST /blob` to upload a new image to local storage, and get a URL.


### Authentication

The token is passed either among query parameters (`/txt2img?token=...`), or via the `Authorization` header 
as a `Bearer` token [(OAuth2 Bearer Authentication)](https://swagger.io/docs/specification/authentication/bearer-authentication/).

To disable authentication and allow generation of public tokens at `POST /token/all`,
set environment variable `ENABLE_PUBLIC_ACCESS=1`.

To allow users to sign up at `POST /user`, 
set environment variable `ENABLE_SIGNUP=1`. 
Registered users can generate their own tokens at `POST /token/{username}`.

### Synchronous Interface

For convenience, the API provides synchronous endpoints at `GET /txt2img`, `GET /img2img`, and `GET /inpaint`.

To print a browser-accessible URL upon startup (i.e., `http://localhost:8000/txt2img?prompt=corgi&steps=5?token=...`), 
set environment variable `PRINT_LINK_WITH_TOKEN=1` (set by default in `.env.example`).

If the connection is dropped (i.e., you navigate away from the page),
the API will automatically cancel the request and free up resources.

It is preferable to use the asynchronous interface for production use.

### Asynchronous Interface

`POST /task` with either `Txt2ImgParams`, `Img2ImgParams` or `InpaintParams` to start a task, and get a `task_id`. 

`GET /task/{task_id}` to get the last `event` broadcast by the task, or subscribe to the websocket endpoint `/events?token=<token>` to get a stream of events as they occur.

Event types:
- PendingEvent
- StartedEvent
- FinishedEvent (with `blob_url` and `parameters_used`)
- AbortedEvent (with `reason`)

To cancel a task, `DELETE /task/{task_id}`.

## Running

### Installing

Install a virtual environment with python 3.10 and poetry.

#### Conda (for example)

Setup [MiniConda](https://docs.conda.io/en/latest/miniconda.html) and create a new environment with python 3.10.

```bash
conda create -n sda python=3.10
conda activate sda
```

#### Poetry

Setup [Poetry](https://python-poetry.org/docs/#installation) and install the dependencies.

```bash
poetry install
```

### Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Genereate a new `SECRET_KEY`, and replace the one from the example:

```bash
openssl rand -hex 32
```

The various environment variables are:

- `SECRET_KEY`: The secret key used to sign the JWT tokens.
- `PRINT_LINK_WITH_TOKEN`: Whether to print a link with the token to the console on startup.
- `ENABLE_PUBLIC_ACCESS`: Whether to enable public token generation (anything except empty string enables it).
- `ENABLE_SIGNUP`: Whether to enable user signup (anything except empty string enables it).
- `BASE_URL`: Used to build link with token printed upon startup and local storage blob URLs.
- `REDIS_HOST`: The host of the Redis server.
- `REDIS_PORT`: The port of the Redis server.
- `REDIS_PASSWORD`: The password of the Redis server.
- `HUGGINGFACE_TOKEN`: The token used by the worker to access the Hugging Face API.

### Docker Compose

Run the API and five workers, with redis as intermediary, and docker compose to manage the containers.

```bash
make run
```

### Multi Process

Or invoke processes on multiple machines, starting the API with:

```bash
poetry run uvicorn stable_diffusion_api.api.redis_app:app
```

And the worker(s) with:

```bash
poetry run python -m stable_diffusion_api.engine.worker.redis_worker
```

### Single Process

Or run the API and worker in a single process.

```bash
poetry run uvicorn stable_diffusion_api.api.in_memory_app:app
```