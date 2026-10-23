Single-machine test with a smaller GLM-5.3 quant (two stage servers, one GPU, two ports):
  stage A: layers [0, K)  embed = true  head = false  listen 127.0.0.1:7801
  stage B: layers [K, N)  embed = false head = true   listen 127.0.0.1:7802   (both with model_dir = the small quant)
  pipeline.toml with both addresses, then:  swp-exl3 generate --pipeline pipeline.toml --prompt "Hello" --chat --bench
Then change K, restart the stages, and compare logits (cut-invariance):
  swp-exl3 logits --pipeline p.toml --prompt "The capital of France is" --save cutA.json     # cut at K1
  swp-exl3 logits --pipeline p.toml --prompt "The capital of France is" --save cutB.json     # cut at K2
  swp-exl3 compare cutA.json cutB.json
