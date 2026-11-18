"""OpenAI-compatible TTS server for nano-edge-tts (Kokoro-82M with a 1.75M-parameter decoder).

    uv run server.py      # 127.0.0.1:8020, 4 workers; override with HOST, PORT, WORKERS
"""
import io
import logging
import os
import threading
import time
import warnings
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before torch loads: one thread per worker is fastest (see README)

import numpy as np
import uvicorn
from rich.logging import RichHandler
from rich.markup import escape
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import hf_hub_download, list_repo_files
from pydantic import BaseModel, Field

SR = 24000
KOKORO = "hexgrad/Kokoro-82M"
NANO = "drmhse/nano-edge-tts"
MAX_CHARS = 20_000
MEDIA = {"wav": "audio/wav", "flac": "audio/flac", "pcm": "audio/pcm"}

SENTENCES = r"(?<=[.!?…])\s+|\n+"  # stream sentence by sentence; Kokoro's default splits only on newlines

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]",
                    handlers=[RichHandler(show_path=False, rich_tracebacks=True)])
logging.getLogger("httpx2").setLevel("WARNING")  # huggingface_hub logs every HEAD request otherwise
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel("ERROR")  # "not supported in MacOs"
warnings.filterwarnings("ignore", module="torch")  # deprecation chatter from Kokoro's model code
LOG = logging.getLogger("nano_edge")
RICH = {"markup": True, "highlighter": None}  # log lines that colour themselves

state = {}
# ponytail: one model call at a time, taken per chunk (never across a yield) so concurrent streams interleave;
# run more workers if you need real parallelism
lock = threading.Lock()


def parse_voice(spec: str) -> list[tuple[str, float]]:
    """'af_heart' | 'af_heart,af_bella' (equal mix) | 'af_heart:0.7,af_bella:0.3' -> [(name, weight)], weights summing to 1."""
    parts = []
    for item in spec.split(","):
        name, _, w = item.strip().partition(":")
        try:
            weight = float(w) if w else 1.0
        except ValueError:
            raise ValueError(f"bad weight in {item!r}")
        if not name or weight <= 0:
            raise ValueError(f"bad voice entry {item!r}")
        parts.append((name, weight))
    total = sum(w for _, w in parts)
    return [(n, w / total) for n, w in parts]


@asynccontextmanager
async def lifespan(app):
    from kokoro import KPipeline
    from nano_edge_tts import load

    for f in ("config.json", "nano-edge-tts.safetensors"):
        weights = hf_hub_download(NANO, f)
    model = load(folder=os.path.dirname(weights), repo_id=KOKORO)
    # 'a' = American English voices (af_/am_), 'b' = British (bf_/bm_); both share the one model
    state["pipes"] = {lang: KPipeline(lang_code=lang, model=model, repo_id=KOKORO) for lang in "ab"}
    state["voices"] = sorted(f[7:-3] for f in list_repo_files(KOKORO)
                             if f.startswith("voices/") and f.endswith(".pt") and f[7] in "ab")
    LOG.info(f"[bold green]Ready[/] · worker {os.getpid()} · {len(state['voices'])} voices · "
             f"{os.environ['OMP_NUM_THREADS']} thread(s)", extra=RICH)
    yield


app = FastAPI(title="nano-edge-tts server", lifespan=lifespan)


@lru_cache(maxsize=256)
def voice_tensor(spec: str):
    parts = parse_voice(spec)
    unknown = [n for n, _ in parts if n not in state["voices"]]
    if unknown:
        raise ValueError(f"unknown voice(s): {', '.join(unknown)}; see GET /v1/voices")
    pipe = state["pipes"][parts[0][0][0]]
    return pipe, sum(w * pipe.load_voice(n) for n, w in parts)


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=MAX_CHARS)
    voice: str = "af_heart"
    model: str = "nano-edge-tts"  # accepted for OpenAI client compatibility, ignored
    speed: float = Field(1.0, ge=0.25, le=4.0)
    response_format: Literal["wav", "flac", "pcm"] = "wav"
    stream: bool = False  # send audio sentence by sentence as it is generated (wav or pcm)


def chunks(pipe, voice, req, split_pattern=r"\n+"):
    """Audio chunks as float32 arrays, holding the model lock only while each one is generated."""
    it = pipe(req.input, voice=voice, speed=req.speed, split_pattern=split_pattern)
    while True:
        with lock, torch.inference_mode():
            r = next(it, None)
        if r is None:
            return
        if r.audio is not None:
            yield r.audio.numpy()


def pcm16(audio):  # raw 16-bit little-endian mono, 24 kHz
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def wav_header_unknown_length():
    # RIFF/data sizes of 0xFFFFFFFF mean "until the stream ends"; browsers, ffmpeg, mpv and sox accept it
    return (b"RIFF\xff\xff\xff\xffWAVEfmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
            + (1).to_bytes(2, "little") + SR.to_bytes(4, "little") + (SR * 2).to_bytes(4, "little")
            + (2).to_bytes(2, "little") + (16).to_bytes(2, "little") + b"data\xff\xff\xff\xff")


class ClosingStream(StreamingResponse):
    """Closes the generator once the response ends, so a client hanging up reaches it as GeneratorExit
    (Starlette leaves it paused until garbage collection)."""

    def __init__(self, gen, **kw):
        super().__init__(gen, **kw)
        self.gen = gen

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.gen.close()  # safe: anyio lets the running sentence finish before the cancel lands


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):  # sync def: FastAPI runs it in a threadpool, keeping the event loop free
    tag, started = os.urandom(2).hex(), time.perf_counter()
    LOG.info(f"[dim cyan]{tag}[/] [bold magenta]speech[/] ← {len(req.input)} chars · voice {escape(req.voice)} · "
             f"{req.response_format}" + (" · stream" if req.stream else "")
             + (f" · speed {req.speed:g}x" if req.speed != 1.0 else ""), extra=RICH)

    def done(samples, size, first=None):
        took = time.perf_counter() - started
        LOG.info(f"[dim cyan]{tag}[/] [bold magenta]speech[/] [green]done[/] in {took:.2f} s · "
                 + (f"first audio {first:.2f} s · " if first is not None else "")
                 + f"{samples / SR:.1f} s audio ({samples / SR / took:.0f}x) · {size / 1024:,.0f} KB", extra=RICH)

    def reject(message):
        LOG.warning(f"[dim cyan]{tag}[/] [yellow]✗ 400[/] {escape(message)}", extra=RICH)
        return HTTPException(400, message)

    try:
        pipe, voice = voice_tensor(req.voice)
    except ValueError as e:
        raise reject(str(e))
    if req.stream:
        if req.response_format == "flac":
            raise reject("streaming supports wav and pcm")
        def body():
            samples = size = 0
            first = None
            try:
                if req.response_format == "wav":
                    yield (head := wav_header_unknown_length())
                    size = len(head)
                for audio in chunks(pipe, voice, req, SENTENCES):
                    if first is None:
                        first = time.perf_counter() - started
                    samples, size = samples + len(audio), size + 2 * len(audio)
                    yield pcm16(audio)
            except GeneratorExit:  # the client hung up; generation stops here
                LOG.warning(f"[dim cyan]{tag}[/] [yellow]✗ client left[/] after {time.perf_counter() - started:.2f} s"
                            f" · {samples / SR:.1f} s audio made", extra=RICH)
                raise
            done(samples, size, first)
        return ClosingStream(body(), media_type=MEDIA[req.response_format])
    parts = list(chunks(pipe, voice, req))
    if not parts:
        raise reject("nothing to speak in input")
    audio = np.concatenate(parts)
    if req.response_format == "pcm":
        body = pcm16(audio)
    else:
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format=req.response_format.upper(), subtype="PCM_16")
        body = buf.getvalue()
    done(len(audio), len(body))
    return Response(body, media_type=MEDIA[req.response_format])


@app.get("/v1/voices")
def voices():
    return {"voices": state["voices"],
            "blend": "join voices with commas for an equal mix (af_heart,af_bella) or add weights (af_heart:0.7,af_bella:0.3)"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "nano-edge-tts", "object": "model", "owned_by": "drmhse"}]}


@app.get("/health")
def health():
    return {"ok": "pipes" in state}


if __name__ == "__main__":
    # log_config=None: uvicorn's own logs go through the rich handler above instead of its plain formatter
    uvicorn.run("server:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8020)),
                workers=int(os.environ.get("WORKERS", 4)), log_config=None)
