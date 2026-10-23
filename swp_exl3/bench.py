"""Per-stage time accounting + tokens/s (same definitions as `swp-drive --bench` in splitwire).
compute = stage_ms reported by the stage (device-synced); wire+queue = client round trip - compute."""


class Trace:
    def __init__(self, layers):
        self.layers = layers
        self.hops = []   # (phase, stage, rtt_ms, stage_ms)

    def add(self, phase, stage, rtt_ms, stage_ms):
        self.hops.append((phase, stage, rtt_ms, stage_ms))

    def sums(self, phase, stage):
        c = o = 0.0
        for ph, st, rtt, sm in self.hops:
            if ph == phase and st == stage:
                c += sm
                o += max(rtt - sm, 0.0)
        return c, o

    def report(self, prefill_tokens, prefill_wall_ms, decode_tokens, decode_wall_ms, out=print):
        res = {"prefill_tok_s": prefill_tokens / (prefill_wall_ms / 1e3) if prefill_wall_ms else 0.0,
               "decode_tok_s": decode_tokens / (decode_wall_ms / 1e3) if decode_wall_ms else 0.0, "stages": []}
        out(f"PREFILL {prefill_tokens} tok in {prefill_wall_ms:.1f} ms -> {res['prefill_tok_s']:.1f} tok/s")
        out(f"DECODE  {decode_tokens} tok in {decode_wall_ms:.1f} ms -> {res['decode_tok_s']:.2f} tok/s")
        out("  stage layers   prefill: compute  wire+q   decode/tok: compute  wire+q   busy")
        nd = max(decode_tokens, 1)
        for i, (a, b) in enumerate(self.layers):
            pc, po = self.sums("prefill", i)
            dc, do = self.sums("decode", i)
            res["stages"].append({"stage": i, "layers": [a, b], "prefill_compute_ms": pc, "prefill_wire_ms": po,
                                  "decode_compute_ms_per_tok": dc / nd, "decode_wire_ms_per_tok": do / nd})
            busy = dc / decode_wall_ms * 100 if decode_wall_ms else 0
            out(f"  {i:>5} {a:>2}..{b:<2}   {pc:>16.1f} {po:>7.1f}   {dc/nd:>18.3f} {do/nd:>7.3f}   {busy:>3.0f}%")
        return res
