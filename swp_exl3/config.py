"""TOML configuration for stage servers and the pipeline coordinator."""
import os, tomllib
from dataclasses import dataclass, field


class ConfigError(Exception):
    pass


def _secret(secret_env, insecure):
    if secret_env:
        v = os.environ.get(secret_env)
        if not v:
            raise ConfigError(f"environment variable {secret_env} (secret_env) is not set")
        return v
    if not insecure:
        raise ConfigError("set secret_env = \"VARNAME\" (recommended) or insecure = true")
    return None


@dataclass
class StageConfig:
    node: str
    listen: str
    model_dir: str
    layers: tuple
    embed: bool
    head: bool
    device: str = "cuda:0"
    cache_tokens: int = 8192
    max_chunk: int = 512
    wire_dtypes: tuple = ("f32", "bf16")
    secret_env: str = None
    insecure: bool = False
    recurrent_dtype: str = "half"
    max_history: int = 0
    spec_k: int = 3                                 # speculative draft window; 0 disables the `spec` capability
    warmup: bool = True                             # synthetic forwards after load() to remove first-prefill warm-up
    engine: dict = field(default_factory=dict)     # set on config.infer_params, e.g. moe_cpu_offload
    load: dict = field(default_factory=dict)       # extra kwargs for model.load()
    env: dict = field(default_factory=dict)        # environment variables set before importing exllamav3

    def secret(self):
        return _secret(self.secret_env, self.insecure)


@dataclass
class PipelineConfig:
    stages: list                                   # [(addr, (a, b))] in pipeline order
    model_dir: str = None                          # for tokenizer / contract on the coordinator
    secret_env: str = None
    insecure: bool = False
    wire_dtype: str = "f32"
    chunk: int = 512
    topk: int = 64
    stage_configs: list = field(default_factory=list)   # parallel to stages: local stage.toml path or None

    def secret(self):
        return _secret(self.secret_env, self.insecure)


def _layers(v, where):
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v) and 0 <= v[0] < v[1]):
        raise ConfigError(f"{where}: layers must be [first, end) with 0 <= first < end, got {v!r}")
    return tuple(v)


def load_stage(path):
    with open(path, "rb") as f:
        t = tomllib.load(f).get("stage") or {}
    for k in ("node", "listen", "model_dir", "layers"):
        if k not in t:
            raise ConfigError(f"[stage] is missing '{k}'")
    c = StageConfig(
        node=t["node"], listen=t["listen"], model_dir=t["model_dir"], layers=_layers(t["layers"], "[stage]"),
        embed=bool(t.get("embed", False)), head=bool(t.get("head", False)), device=t.get("device", "cuda:0"),
        cache_tokens=int(t.get("cache_tokens", 8192)), max_chunk=int(t.get("max_chunk", 512)),
        wire_dtypes=tuple(t.get("wire_dtypes", ("f32", "bf16"))), secret_env=t.get("secret_env"),
        insecure=bool(t.get("insecure", False)), recurrent_dtype=t.get("recurrent_dtype", "half"),
        max_history=int(t.get("max_history", 0)), spec_k=int(t.get("spec_k", 3)),
        warmup=bool(t.get("warmup", True)),
        engine=dict(t.get("engine", {})), load=dict(t.get("load", {})), env=dict(t.get("env", {})))
    if c.cache_tokens % 256:
        raise ConfigError("cache_tokens must be a multiple of 256")
    return c


def load_pipeline(path):
    with open(path, "rb") as f:
        d = tomllib.load(f)
    p, st = d.get("pipeline") or {}, d.get("stage") or []
    if not st:
        raise ConfigError("no [[stage]] entries")
    stages = [(s.get("addr") or "local", _layers(s["layers"], f"[[stage]] {s.get('addr')}")) for s in st]
    for i, s in enumerate(st):
        if not s.get("addr") and not s.get("stage_config"):
            raise ConfigError(f"[[stage]] {i}: set addr (remote) or stage_config (in-process)")
    for (a0, (_, e0)), (a1, (s1, _)) in zip(stages, stages[1:]):
        if e0 != s1:
            raise ConfigError(f"stage layer ranges must be contiguous: {a0} ends at {e0} but {a1} starts at {s1}")
    if stages[0][1][0] != 0:
        raise ConfigError("the first stage must start at layer 0")
    return PipelineConfig(stages=stages, model_dir=p.get("model_dir"), secret_env=p.get("secret_env"),
                          insecure=bool(p.get("insecure", False)), wire_dtype=p.get("wire_dtype", "f32"),
                          chunk=int(p.get("chunk", 512)), topk=int(p.get("topk", 64)),
                          stage_configs=[s.get("stage_config") for s in st])
