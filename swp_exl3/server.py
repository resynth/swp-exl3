"""SWP/1 stage server: one loaded slice, one active session, any number of connections (calls are serialized)."""
import os, socket, threading, time, traceback
import numpy as np
from . import swp
from .backend import Unsupported
from .contract import contract_hash, diff_contracts

DTYPE_BYTES = {"f32": 4, "bf16": 2}


class LocalPeer:
    """In-process replacement for `swp.Peer`: calls the stage server directly (no socket, no framing)."""

    def __init__(self, server, node="coordinator"):
        self.server, self.node, self.seq, self.addr = server, node, 0, "local"

    def rpc(self, kind, header, payload=b"", want=None):
        self.seq += 1
        frame = self.server.handle(kind, self.seq, header, payload)
        return swp.expect(frame, want)

    def ping(self, payload=b""):
        return self.rpc(swp.PING, {}, payload, swp.PONG)

    def close(self):
        pass


class StageServer:
    def __init__(self, backend, node, listen, secret=None, wire_dtypes=("f32", "bf16"), max_chunk=2048, log=print):
        self.b, self.node, self.listen, self.secret = backend, node, listen, secret
        self.wire_dtypes, self.max_chunk, self.log = tuple(wire_dtypes), max_chunk, log
        self.lock = threading.Lock()
        self.session = None          # id of the configured session; None => must (re)configure
        self.wire = "f32"
        self.caps = ["stage", "topk"] + [f"wire:{d}" for d in self.wire_dtypes]
        self.debug = bool(os.environ.get("SWP_DEBUG_TIMING"))

    # ---- lifecycle
    def bind(self):
        host, port = self.listen.rsplit(":", 1)
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        return self.port

    def serve_forever(self):
        if not hasattr(self, "sock"):
            self.bind()
        self.log(f"[{self.node}] layers {self.b.layers} listening on :{self.port} ({'authenticated' if self.secret else 'INSECURE'})")
        while True:
            try:
                c, addr = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._conn, args=(c, addr), daemon=True).start()

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    # ---- connection
    def _conn(self, c, addr):
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            swp.server_handshake(c, self.node, self.caps, self.secret)
            while True:
                t0 = time.perf_counter()
                kind, seq, h, payload = swp.recv_frame(c)
                t1 = time.perf_counter()
                try:
                    out = self.handle(kind, seq, h, payload, t0, t1)
                except swp.SwpError as e:
                    out = (swp.ERROR, seq, {"code": e.code, "message": e.message}, b"")
                t2 = time.perf_counter()
                swp.send_frame(c, out[0], out[1], out[2], out[3])
                if self.debug:
                    t3 = time.perf_counter()
                    self.log(f"[srv] kind={kind:#x} n={len(payload)}B recv={(t1-t0)*1e3:.3f} "
                             f"handle={(t2-t1)*1e3:.3f} send={(t3-t2)*1e3:.3f}", flush=True)
        except (ConnectionError, OSError):
            pass
        except swp.SwpError as e:
            self.log(f"[{self.node}] {addr}: {e}")
        except Exception:
            self.log(f"[{self.node}] {addr}: unexpected error\n{traceback.format_exc()}")
        finally:
            c.close()

    def handle(self, kind, seq, h, payload, t_recv_start=None, t_recv_done=None):
        """Process one request and return (out_kind, seq, out_header, out_payload); raises swp.SwpError."""
        if kind == swp.PING:
            return swp.PONG, seq, h, payload
        with self.lock:
            if kind == swp.STAGE_CONFIGURE:
                return self._configure(seq, h)
            if kind == swp.FORWARD:
                return self._forward(seq, h, payload, t_recv_start, t_recv_done)
            if kind == swp.CACHE_OP:
                return self._cache_op(seq, h)
            if kind == swp.SESSION_CLOSE:
                self.b.reset()
                self.session = None
                return swp.SESSION_CLOSED, seq, {"session": h.get("session")}, b""
        raise swp.SwpError("unsupported", f"kind {kind:#x}")

    # ---- handlers
    def _configure(self, seq, h):
        b = self.b
        want = (tuple(h.get("layers", ())), bool(h.get("has_embed")), bool(h.get("has_head")))
        have = (tuple(b.layers), b.has_embed, b.has_head)
        if want != have:
            raise swp.SwpError("contract_mismatch", f"this stage holds layers {have[0]} embed={have[1]} head={have[2]}; asked for {want}")
        wd = h.get("wire_dtype", "f32")
        if wd not in self.wire_dtypes:
            raise swp.SwpError("unsupported", f"wire dtype {wd}; this stage supports {self.wire_dtypes}")
        if h.get("contract"):
            common = {k: v for k, v in h["contract"].items() if k in b.contract}
            d = diff_contracts(common, {k: b.contract[k] for k in common})
            if d:
                raise swp.SwpError("contract_mismatch", "; ".join(d))
        self.b.reset()
        self.session, self.wire = h.get("session", "s"), wd
        return swp.STAGE_READY, seq, {"session": self.session, "cache_len": 0, "tail": list(b.tail),
                                      "contract_hash": contract_hash(b.contract), "memory": getattr(b, "memory", {})}, b""

    def _forward(self, seq, h, payload, t_recv_start=None, t_recv_done=None):
        b = self.b
        if self.session is None or h.get("session") != self.session:
            raise swp.SwpError("no_session", "configure the session first (stage may have been reset after an error)")
        n, start = int(h["n_rows"]), int(h["start_pos"])
        toks = [int(t) for t in h.get("token_ids", [])]
        if n < 1 or n > self.max_chunk or len(toks) != n:
            raise swp.SwpError("bad_request", f"n_rows={n}, token_ids={len(toks)}, max_chunk={self.max_chunk}")
        if start != b.cache_len():
            if h.get("rewrite") and start < b.cache_len():
                try:
                    b.truncate(start)
                except Unsupported as e:
                    raise swp.SwpError("unsupported", str(e))
            else:
                raise swp.SwpError("pos_mismatch", f"expected start_pos {b.cache_len()} got {start}")
        hidden = None
        if h.get("input") == "hidden":
            need = n * int(np.prod(b.tail)) * DTYPE_BYTES[self.wire]
            if len(payload) != need:
                raise swp.SwpError("bad_request", f"hidden payload {len(payload)} bytes, expected {need} ({n} x {b.tail} {self.wire})")
            hidden = payload
        elif not b.has_embed:
            raise swp.SwpError("bad_request", "this stage has no embedding: send input=hidden")
        t_pre = time.perf_counter()
        try:
            res = b.forward(start, toks, hidden, self.wire, h.get("want", "token"), h.get("return_rows", "last"), h.get("k"))
        except Unsupported as e:
            raise swp.SwpError("unsupported", str(e))
        except Exception as e:
            self.session = None          # engine state is unknown after a failure: force reconfigure + replay
            code = "out_of_memory" if "OutOfMemory" in type(e).__name__ else "engine_error"
            self.log(f"[{self.node}] forward failed: {traceback.format_exc()}")
            raise swp.SwpError(code, f"{type(e).__name__}: {e}")
        wall = (time.perf_counter() - t_pre) * 1e3
        timing = dict(res.timing)
        if t_recv_done is not None:
            # srv_recv_ms includes the wait for the request, i.e. the time the other stage was computing.
            timing["srv_recv_ms"] = (t_recv_done - t_recv_start) * 1e3
            timing["srv_pre_ms"] = (t_pre - t_recv_done) * 1e3
        out = {"session": self.session, "step": h.get("step"), "kind": res.kind, "dtype": res.dtype, "shape": res.shape,
               "stage_ms": timing.get("compute_ms", wall), "timing": dict(timing, total_ms=wall)}
        return swp.FORWARD_OUT, seq, out, res.payload

    def _cache_op(self, seq, h):
        op = h.get("op")
        try:
            if op == "truncate":
                self.b.truncate(int(h["pos"]))
            elif op == "clear":
                self.b.reset()
            else:
                raise swp.SwpError("unsupported", f"cache op {op}")
        except Unsupported as e:
            raise swp.SwpError("unsupported", str(e))
        return swp.CACHE_OK, seq, {"session": self.session, "cache_len": self.b.cache_len()}, b""
