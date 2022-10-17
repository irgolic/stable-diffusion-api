# Stable Diffusion Server

[![OpenApi](https://img.shields.io/badge/OpenApi-3.0.2-black)](https://editor.swagger.io/?url=https://raw.githubusercontent.com/irgolic/stable-diffusion-server/master/openapi.yml?token%3DGHSAT0AAAAAABTFSDOFSU2W23KZ4XG72RYGY2MXGZA)

A lightweight server for deploying stable diffusion models. Easily run Txt2Img and Img2Img with any model published on [Hugging Face](https://huggingface.co/models).

## Usage

Generate any client library from the [OpenApi](
https://editor.swagger.io/?url=https://raw.githubusercontent.com/irgolic/stable-diffusion-server/master/openapi.yml?token%3DGHSAT0AAAAAABTFSDOFSU2W23KZ4XG72RYGY2MXGZA) specification.

**TODO**  
_add example with either bash, python, or javascript._

### Authentication

Set environment variable `ENABLE_PUBLIC_TOKEN` to allow generation of public tokens. 
Public tokens are not tied to a user and can be used by anyone.

Alternatively, set environment variable `ENABLE_SIGNUP` to allow users to sign up and generate their own tokens.

### Invocation

`POST /txt2img` or `POST /img2img` to start a task, and get a `task_id`. 
The model will be downloaded and cached, and the task will be queued for execution.

Supported parameters:
- `prompt`: text prompt, e.g. `A cat` **required**
- `negative_prompt`: negative prompt, e.g. `A dog`
- `model_id`: model name, e.g. `CompVis/stable-diffusion-v1-4`
- `model_provider`: model provider, e.g. `huggingface`
- `width`: image width, e.g. `512`
- `height`: image height, e.g. `512`
- `steps`: number of steps, e.g. `20`
- `guidance`: relatedness to `prompt`, e.g. `7.5`
- `scheduler`: image generation schedule r, e.g. `linear`
- `safety_filter`: enable safety checker, e.g. `true`

POST `/txt2img` also supports:
- `width`: image width, e.g. `512`
- `height`: image height, e.g. `512`

POST `/img2img` also supports:
- `initial_image`: image file **required**
- `strength`: how much to change the image, e.g. `0.8`

Img2Img can reference a previously generated image's `blob_id`. 
Alternatively, POST `/blob` to upload a blob, and get a new `blob_id`. 

### Job Status

`GET /task/{task_id}` to get the last `event` fired by the task of a task.

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
- [ ] Seed parameter

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