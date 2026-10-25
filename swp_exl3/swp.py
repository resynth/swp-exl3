"""SWP/1 transport: framing, handshake (client and server), RPC helpers. Pure stdlib + `blake3` (auth/hash only).
Wire format: see the SWP/1 spec doc, section A3. Interoperates with the Rust reference tools (splitwire).

Canonical copy: `splitwire/python/swp.py`. This file is vendored verbatim into `swp_exl3/swp.py` so the
adapter stays installable on its own; edit the canonical copy in splitwire and re-copy, and
`tests/test_swp_sync.py` fails if the two diverge.

Speculative verification (additive, capability `spec`), SWP/1 rev 2
-------------------------------------------------------------------
A stage advertises `spec` in its HELLO/HELLO_OK capabilities and `spec_k` (max draft tokens) in
STAGE_READY. A FORWARD may then carry:

    mode        "verify" (default when absent: "prefill"/"decode" as today)
    draft_ids   [d_1 .. d_k], the proposed continuation; n_rows == 1 + len(draft_ids) and
                token_ids == [t] + draft_ids (t = last accepted token, not yet fed)
    accept      "argmax" | "sample" (how the head stage decides acceptance)
    sampling    {"temperature", "top_p", "top_k", "seed", "round"} for accept == "sample"
    next_drafts true -> ask a stage with a drafter for the next round's drafts (MTP, optional)
    draft_k     draft window for a normal (non-verify) forward that asks for next_drafts

Every stage that handles a verify forward records recurrent history (`last_history = n_rows - 1`),
so the next forward may rewind up to that many rows by sending `start_pos < cache_len` with
`rewrite: true`. A rewind is legal only immediately after a verify forward; a stage that cannot
honour it answers ERROR `unsupported` and the coordinator clears + replays.

The head stage may answer `kind: "spec"` (FORWARD_OUT) with

    accepted    m, number of leading drafts the target agreed with
    tokens      [d_1 .. d_m, bonus] (payload: int32, same order)
    drafts      next round's drafts (empty unless a drafter is attached)
    drafts_basis the token the returned drafts follow
    timing      the usual stage timing dict

A stage advertising the additive capability `mtp` may also attach `drafts` / `drafts_basis` to a
normal prefill/decode reply when the FORWARD carries `next_drafts: true` and `draft_k: k`.

so one verify of k+1 rows advances the sequence by m+1 tokens. `kind: "topk"` with
`return_rows: "all"` is the same verify without acceptance, used by the logit-equivalence check.
"""
import json, os, socket, struct

MAGIC = b"SWP1"
HELLO, HELLO_OK, ERROR, PING, PONG = 0x01, 0x02, 0x0F, 0x10, 0x11
STAGE_CONFIGURE, STAGE_READY, FORWARD, FORWARD_OUT = 0x20, 0x21, 0x22, 0x23
CACHE_OP, CACHE_OK, SESSION_CLOSE, SESSION_CLOSED = 0x24, 0x25, 0x26, 0x27
MANIFEST_GET, MANIFEST, TENSOR_GET, TENSOR_DATA, AUX_GET, AUX_DATA = 0x40, 0x41, 0x42, 0x43, 0x44, 0x45
FLAG_HASH = 1
PROTO = "swp/1"
CAP_SPEC = "spec"                 # speculative verification; see the module docstring
CAP_MTP = "mtp"                   # head-stage MTP drafter can supply next-round drafts
MAX_HEADER, MAX_PAYLOAD = 1 << 20, 8 << 30


class SwpError(Exception):
    def __init__(self, code, message=""):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        r = sock.recv_into(view[got:], min(n - got, 1 << 22))
        if r == 0:
            raise ConnectionError("peer closed")
        got += r
    return buf


def _send_iov(sock, bufs):
    """One sendmsg for the common case; loops only if the kernel takes a short write."""
    mvs = [memoryview(b) for b in bufs if len(b)]
    i = 0
    while i < len(mvs):
        n = sock.sendmsg(mvs[i:])
        if n <= 0:
            raise ConnectionError("send failed")
        while i < len(mvs) and n >= mvs[i].nbytes:
            n -= mvs[i].nbytes
            i += 1
        if i < len(mvs) and n:
            mvs[i] = mvs[i][n:]


def send_frame(sock, kind, seq, header, payload=b"", hash=False):
    hdr = json.dumps(header, separators=(",", ":")).encode()
    if len(hdr) > MAX_HEADER:
        raise SwpError("bad_request", "header too large")
    bufs = [MAGIC + struct.pack("<BBHIIQ", kind, FLAG_HASH if hash else 0, 0, seq, len(hdr), len(payload)), hdr, payload]
    if hash:
        import blake3
        bufs.append(blake3.blake3(bytes(payload)).digest()[:16])
    _send_iov(sock, bufs)


def recv_frame(sock):
    fixed = _recv_exact(sock, 24)
    if bytes(fixed[:4]) != MAGIC:
        raise ConnectionError("bad magic (peer is not speaking SWP/1)")
    kind, flags, _r, seq, hl, pl = struct.unpack("<BBHIIQ", bytes(fixed[4:]))
    if hl > MAX_HEADER or pl > MAX_PAYLOAD:
        raise ConnectionError("frame too large")
    header = json.loads(bytes(_recv_exact(sock, hl))) if hl else {}
    payload = _recv_exact(sock, pl) if pl else bytearray()
    if flags & FLAG_HASH:
        import blake3
        if bytes(_recv_exact(sock, 16)) != blake3.blake3(bytes(payload)).digest()[:16]:
            raise ConnectionError("payload hash mismatch")
    return kind, seq, header, payload


def proof(secret, nonce, node):
    import blake3
    key = blake3.blake3(secret.encode()).digest()
    return blake3.blake3(f"{nonce}|{node}".encode(), key=key).hexdigest()


def server_handshake(sock, node, caps, secret=None):
    """Server speaks first. Returns the client's HELLO header. Challenge-response only; NOT encryption."""
    nonce = os.urandom(16).hex()
    send_frame(sock, HELLO, 0, {"proto": PROTO, "node": node, "caps": list(caps), "nonce": nonce, "auth": secret is not None})
    kind, seq, h, _ = recv_frame(sock)
    if kind != HELLO or h.get("proto") != PROTO:
        send_frame(sock, ERROR, seq, {"code": "proto", "message": "expected swp/1 HELLO"})
        raise SwpError("proto", "bad client hello")
    if secret is not None and h.get("proof") != proof(secret, nonce, h.get("node", "")):
        send_frame(sock, ERROR, seq, {"code": "auth", "message": "authentication failed"})
        raise SwpError("auth", "authentication failed")
    send_frame(sock, HELLO_OK, seq, {"proto": PROTO, "node": node, "caps": list(caps)})
    return h


def client_handshake(sock, node, caps, secret=None):
    kind, _s, sh, _p = recv_frame(sock)
    if kind != HELLO or sh.get("proto") != PROTO:
        raise SwpError("proto", "not an swp/1 server")
    hello = {"proto": PROTO, "node": node, "caps": list(caps)}
    if sh.get("auth"):
        if not secret:
            raise SwpError("auth", "server requires a secret (set the env var named by secret_env)")
        hello["proof"] = proof(secret, sh["nonce"], node)
    send_frame(sock, HELLO, 1, hello)
    k, _s, h, _p = recv_frame(sock)
    if k == ERROR:
        raise SwpError(h.get("code", "error"), h.get("message", ""))
    if k != HELLO_OK:
        raise ConnectionError("unexpected handshake reply")
    return sh


def expect(frame, want):
    kind, _seq, header, payload = frame
    if kind == ERROR:
        raise SwpError(header.get("code", "error"), header.get("message", ""))
    if kind != want:
        raise ConnectionError(f"unexpected frame kind {kind:#x}")
    return header, payload


class Peer:
    """Client connection with RPC helpers."""

    def __init__(self, addr, node="swp-exl3", secret=None, caps=("coordinator",), timeout=None):
        host, port = addr.rsplit(":", 1)
        self.addr = addr
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self.seq = 10
        self.server = client_handshake(self.sock, node, caps, secret)

    def rpc(self, kind, header, payload=b"", want=None):
        self.seq += 1
        send_frame(self.sock, kind, self.seq, header, payload)
        return expect(recv_frame(self.sock), want)

    def ping(self, payload=b""):
        return self.rpc(PING, {}, payload, PONG)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class StageClient:
    """Client with the stage RPCs (configure/forward/truncate/close); kept for the splitwire tools."""

    def __init__(self, addr, node="py-client", secret=None, caps=("coordinator",), timeout=None):
        host, port = addr.rsplit(":", 1)
        self.addr = addr
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.seq = 10
        self.server = client_handshake(self.sock, node, caps, secret)

    def _rpc(self, kind, header, payload=b"", want=None):
        self.seq += 1
        send_frame(self.sock, kind, self.seq, header, payload)
        return expect(recv_frame(self.sock), want)

    def configure(self, session, layers, has_embed, has_head, model, wire_dtype="f32"):
        return self._rpc(STAGE_CONFIGURE, {"session": session, "layers": list(layers), "has_embed": has_embed,
                                           "has_head": has_head, "wire_dtype": wire_dtype, "model": model}, want=STAGE_READY)[0]

    def forward(self, header, payload=b""):
        return self._rpc(FORWARD, header, payload, want=FORWARD_OUT)

    def truncate(self, session, pos):
        return self._rpc(CACHE_OP, {"session": session, "op": "truncate", "pos": pos}, want=CACHE_OK)[0]

    def close(self, session):
        return self._rpc(SESSION_CLOSE, {"session": session}, want=SESSION_CLOSED)[0]
