"""Weight provisioning: pull exactly one stage's tensors from a `swp-weightd serve` into a local safetensors shard.

Selection is architecture-aware for GLM-5.3-Flash style checkpoints (key_prefix `model.language_model`, top-level
`hc_head`, final `model.language_model.norm`, MTP layer at index >= num_hidden_layers, optional vision tower).
Output is a normal model directory: engines load it unchanged."""
import json, re, struct
from . import swp

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
VISION_RE = re.compile(r"(?:^|\.)(visual|vision_tower|vision_model|mm_projector|multi_modal_projector)(?:\.|$)")
HEAD_RE = re.compile(r"(?:^|\.)(lm_head|hc_head|norm|output_norm|output)(?:\.|$)")


def classify(name, n_layers):
    """-> (group, layer). groups: layer | extra | embed | head | vision | other"""
    if VISION_RE.search(name):
        return "vision", None
    m = LAYER_RE.search(name)
    if m:
        l = int(m.group(1))
        return ("extra" if n_layers is not None and l >= n_layers else "layer"), l
    if "embed_tokens" in name or "token_embd" in name:
        return "embed", None
    if HEAD_RE.search(name):
        return "head", None
    return "other", None


def select(manifest, a, b, first, last, include_extra=None, include_vision=False, tied_embed=False):
    """Tensors for layers [a, b). `last` stage also gets the MTP/extra layers (include_extra defaults to `last`) and,
    when the checkpoint ties lm_head to the embedding, a copy of the embedding. Unknown tensors go to the first stage."""
    include_extra = last if include_extra is None else include_extra
    n = manifest.get("n_layers")
    out, unknown = [], []
    for t in manifest["tensors"]:
        g, l = classify(t["name"], n)
        if g == "layer":
            keep = a <= l < b
        elif g == "extra":
            keep = include_extra
        elif g == "embed":
            keep = first or (last and tied_embed)
        elif g == "head":
            keep = last
        elif g == "vision":
            keep = include_vision and first
        else:
            keep = first
            if keep:
                unknown.append(t["name"])
        if keep:
            out.append(t)
    return out, unknown


def shard_header(tensors, meta):
    order = sorted(tensors, key=lambda t: t["name"])
    hdr, cur = {"__metadata__": meta}, 0
    for t in order:
        hdr[t["name"]] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [cur, cur + t["len"]]}
        cur += t["len"]
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h, order


class WeightClient:
    def __init__(self, addr, secret=None, node="pull"):
        self.peer = swp.Peer(addr, node=node, secret=secret, caps=("weights-client",))

    def manifest(self):
        _, payload = self.peer.rpc(swp.MANIFEST_GET, {}, want=swp.MANIFEST)
        return json.loads(bytes(payload))

    def aux(self, name):
        _, payload = self.peer.rpc(swp.AUX_GET, {"name": name}, want=swp.AUX_DATA)
        return bytes(payload)

    def fetch_tensor(self, name, out):
        import blake3
        p = self.peer
        p.seq += 1
        swp.send_frame(p.sock, swp.TENSOR_GET, p.seq, {"name": name})
        h, total = blake3.blake3(), 0
        while True:
            hd, payload = swp.expect(swp.recv_frame(p.sock), swp.TENSOR_DATA)
            h.update(payload)
            out.write(payload)
            total += len(payload)
            if hd.get("last"):
                local = h.hexdigest()
                if hd.get("hash") and hd["hash"] != local:
                    raise swp.SwpError("hash_mismatch", f"{name}: transfer hash mismatch")
                return total, local


def pull(server, out_dir, a, b, first, last, secret=None, include_extra=None, include_vision=False, tied_embed=False,
         log=print):
    import os
    c = WeightClient(server, secret)
    m = c.manifest()
    if m.get("format") != "safetensors":
        raise ValueError("pull supports safetensors sources only")
    sel, unknown = select(m, a, b, first, last, include_extra, include_vision, tied_embed)
    if unknown:
        log(f"[pull] {len(unknown)} unclassified tensors assigned to the first stage, e.g. {unknown[:3]}")
    total = sum(t["len"] for t in sel)
    log(f"[pull] {len(sel)} tensors, {total/1e9:.2f} GB for layers {a}..{b} first={first} last={last}")
    os.makedirs(out_dir, exist_ok=True)
    name = f"swp-shard-L{a}-{b}{'-first' if first else ''}{'-last' if last else ''}.safetensors"
    header, order = shard_header(sel, {"swp_source": m.get("manifest_hash"), "swp_layers": f"{a}..{b}", "format": "pt"})
    part = os.path.join(out_dir, name + ".part")
    with open(part, "wb") as f:
        f.write(header)
        for t in order:
            n, h = c.fetch_tensor(t["name"], f)
            if n != t["len"] or (t.get("hash") and t["hash"] != h):
                raise swp.SwpError("hash_mismatch", f"{t['name']}: size/hash differs from the manifest")
    os.replace(part, os.path.join(out_dir, name))
    for af in m.get("aux_files", []):
        p = os.path.join(out_dir, af["name"])
        if not (os.path.exists(p) and os.path.getsize(p) == af["len"]):
            with open(p, "wb") as f:
                f.write(c.aux(af["name"]))
    log(f"[pull] wrote {os.path.join(out_dir, name)}")
    return os.path.join(out_dir, name)
