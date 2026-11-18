# nano-edge-server

OpenAI-compatible text-to-speech server for [nano-edge-tts](https://huggingface.co/drmhse/nano-edge-tts):
Kokoro-82M with its decoder swapped for a 1.75M-parameter one (about 85x real time on an M1 with the default 4 workers).

```sh
uv run --env-file .env uvicorn server:app
```

`.env` sets the defaults: `127.0.0.1:8020`, 4 workers, 1 thread each (see [Throughput](#throughput)).
Command-line flags override it, e.g. `uv run --env-file .env uvicorn server:app --workers 1` while developing.

The first start downloads Kokoro (~330 MB) and the nano-edge weights (7 MB) to the Hugging Face cache.

## Use

```sh
curl localhost:8020/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Rain hammered the ruined watchtower.", "voice": "af_heart"}' -o out.wav
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key="unused")
audio = client.audio.speech.create(model="nano-edge-tts", voice="af_heart", input="Hello.", response_format="wav")
open("out.wav", "wb").write(audio.content)
```

| field | default | |
|---|---|---|
| `input` | required | up to 20,000 characters |
| `voice` | `af_heart` | a voice from `GET /v1/voices`, or a blend (below) |
| `speed` | `1.0` | 0.25 to 4.0 |
| `response_format` | `wav` | `wav`, `flac`, or `pcm` (raw 16-bit little-endian, mono, 24 kHz) |
| `stream` | `false` | `true` sends audio sentence by sentence as it is generated (`wav` or `pcm`) |
| `model` | | accepted for OpenAI clients, ignored |

### Streaming

With `"stream": true` the first audio arrives in about 0.1 s instead of after the whole input is made
(0.7 s for a 27-second paragraph on an M1). A streamed `wav` has an open-ended header, which browsers,
ffmpeg, mpv and soundfile read fine. Each sentence is generated separately, so the pauses between
sentences are a little longer than without streaming.

```sh
curl -sN localhost:8020/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "First sentence. Second sentence.", "response_format": "pcm", "stream": true}' \
  | ffplay -f s16le -ar 24000 -ch_layout mono -nodisp -autoexit -
```

```python
with client.audio.speech.with_streaming_response.create(
    model="nano-edge-tts", voice="af_heart", input=text, response_format="pcm", extra_body={"stream": True}
) as r:
    for chunk in r.iter_bytes():
        play(chunk)  # 16-bit little-endian mono PCM at 24 kHz
```

### Voice blends

- `af_heart,af_bella` is an equal mix.
- `af_heart:0.7,af_bella:0.3` is a weighted mix (weights are normalised).
- A blend uses the accent of its first voice: `a` voices (af_/am_) are American, `b` voices (bf_/bm_) British.

Blends are computed once and cached, so they cost the same as a single voice.

## Endpoints

`POST /v1/audio/speech` · `GET /v1/voices` · `GET /v1/models` · `GET /health`

## Throughput

One worker synthesises one request at a time, so the default `.env` runs 4 single-threaded workers. Send several
requests at once (about 2 per worker) to keep them all busy, for example when rendering a whole book.

Measured on an M1 (4 fast + 4 efficiency cores, 16 GB), rendering a mix of long and short audiobook lines with
twice as many requests in flight as workers:

| workers × threads | speed | RAM |
|---|---|---|
| 1 × 4 (torch default) | 43x real time | 1.1 GB |
| 1 × 1 | 48x | 1.1 GB |
| 2 × 2 | 52x | |
| **4 × 1 (`.env` default)** | **82-86x** | 4.0 GB |
| 6 × 1 | 92x | 4.4 GB |
| 8 × 1 | 86x | |

More threads per worker don't help: the model is too small to split well across cores. Each worker loads its own model.

## Check

```sh
uv run python test_parse_voice.py
```

## Credits

`nano_edge_tts.py` is copied unchanged from [drmhse/nano-edge-tts](https://huggingface.co/drmhse/nano-edge-tts)
(Apache-2.0, see `LICENSE`). Kokoro-82M by [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0.
