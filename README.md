# Stable Diffusion Server

[![OpenApi](https://img.shields.io/badge/OpenApi-3.0.2-black)](https://editor.swagger.io/?url=https://raw.githubusercontent.com/irgolic/stable-diffusion-server/master/openapi.yml)

Easily serve Txt2Img and Img2Img with any model published on [Hugging Face](https://huggingface.co/models).
The aim of this project is to provide a lightweight backend for 

## Usage

Generate any client library from the [OpenApi](
https://editor.swagger.io/?url=https://raw.githubusercontent.com/irgolic/stable-diffusion-server/master/openapi.yml) specification.

### Environment Variables



### Authentication

The server uses [OAuth2 Bearer Token](https://swagger.io/docs/specification/authentication/bearer-authentication/) for authentication. The token is passed in the `Authorization` header as a `Bearer` token.

```http
Authorization: Bearer <token>
```

Set environment variable `ENABLE_PUBLIC_TOKEN` to allow generation of public tokens at `POST /token/all`.

Alternatively, set environment variable `ENABLE_SIGNUP` to allow users to sign up at `POST /user`. 
Registered users can generate their own tokens at `POST /token/{username}`.

### Invocation

`POST /task` with either `Txt2ImgParams` or `Img2ImgParams` to start a task, and get a `task_id`. 
The model will be downloaded and cached, and the task will be queued for execution.

Supported parameters:
- `prompt`: text prompt **(required)**, e.g. `corgi with a top hat`
- `negative_prompt`: negative prompt, e.g. `monocle`
- `model_id`: model name, default `CompVis/stable-diffusion-v1-4`
- `model_provider`: model provider, currently only `huggingface` is supported
- `steps`: number of steps, default `20`
- `guidance`: relatedness to `prompt`, default `7.5`
- `scheduler`: either `plms`, `ddim`, or `k-lms`
- `safety_filter`: enable safety checker, default `true`

Txt2Img also supports:
- `width`: image width, default `512`
- `height`: image height, default `512`

Img2Img also supports:
- `initial_image`: blob id of image to be transformed **(required)**
- `strength`: how much to change the image, default `0.8`

Img2Img can reference a previously generated image's `blob_id`. 
Alternatively, `POST /blob` to upload a new image, and get a `blob_id`.

### Job Status

`GET /task/{task_id}` to get the last `event` broadcast by the task.

Event types:
- PendingEvent
- StartedEvent
- FinishedEvent (with `blob_id`)
- CancelledEvent (with `reason`)

Alternatively, subscribe to the websocket endpoint `/events?token=<token>` to get a stream of events as they occur.

### Results

A FinishedEvent contains an `image` field including its `blob_id` and `parameters_used`.

`GET /blob/{blob_id}` to get the image in PNG format.

## Roadmap

- [ ] GPU support (currently runs only on CPU) – testers needed!
- [ ] Inpainting
- [ ] Seed parameter
- [ ] Cancel task endpoint
- [ ] Progress update events (model download, image generation)
- [ ] Custom tokenizers, supporting `((emphasis))` and `[alternating,prompts,0.4]`
- [ ] More model providers

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