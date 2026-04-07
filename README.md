# Gemma 4 vs Qwen 3.5 — MoE Benchmark on AMD Strix Halo (EVO-X2) with llama.cpp Vulkan

Benchmark comparison of **Gemma 4 26B-A4B** and **Qwen 3.5-35B-A3B** on AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151) using llama.cpp with the Vulkan backend.

Both models are MoE (Mixture-of-Experts) architectures with 3-4B active parameters. This benchmark evaluates them on Japanese text reasoning (JCommonsenseQA), multimodal VLM tasks, and raw inference speed — with NVIDIA DGX Spark results as reference.

## Why Vulkan on Strix Halo?

AMD Strix Halo (gfx1151) has two GPU compute backends available in llama.cpp: **Vulkan (RADV)** and **ROCm (HIP)**. As of April 2026, Vulkan is the better choice for MoE models:

| Aspect | Vulkan (RADV) | ROCm (HIP) |
|---|---|---|
| Build complexity | Easy (`apt install libvulkan-dev`) | Requires ROCm system install |
| MoE tg speed | Faster (recent RADV improvements) | Comparable or slower |
| Stability | Stable | "GGGG" output issues reported |
| GDN support | No shader (CPU fallback) | Kernel exists but inefficient on RDNA 3.5 |

Key references:
- [Vulkan now beats HIP on Strix Halo for MoE models](https://github.com/ggml-org/llama.cpp/discussions/10879) (llama.cpp Discussion #10879)
- [Ubuntu 26.04 brings significant Vulkan gains for Strix Halo](https://www.phoronix.com/review/ubuntu-2604-strix-halo/4) (Phoronix, April 2026)
- [GDN issue on gfx1151](https://github.com/ggml-org/llama.cpp/issues/20354) (llama.cpp Issue #20354)

### The GDN Problem and Why It Matters

Qwen 3.5 uses Gated DeltaNet (GDN), an architecture not yet efficiently supported on AMD hardware. On gfx1151, GDN falls back to CPU in Vulkan and runs at ~12 t/s in HIP — making Dense Qwen 3.5 models (27B) effectively unusable at 1.72 t/s.

**Gemma 4 does not use GDN.** This means Vulkan performance is fully utilized, giving Gemma 4 a structural advantage on Strix Halo that does not exist on NVIDIA hardware.

## Hardware

| Spec | Value |
|---|---|
| CPU/GPU | AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, RDNA 3.5, 40 CUs) |
| Memory | 128 GB LPDDR5X-8000 unified (~256 GB/s bandwidth) |
| VGM (Video Memory) | 64 GB |
| OS | Ubuntu 25.10 (kernel 6.18.20) |
| llama.cpp | b8672 (built from source, Vulkan backend) |
| Vulkan driver | RADV GFX1151 (Mesa) |

### Reference: NVIDIA DGX Spark

| Spec | Value |
|---|---|
| GPU | NVIDIA GB10 (Grace Blackwell, 128 GB unified, ~273 GB/s) |
| llama.cpp | b8665/b8672 (CUDA sm_121) |
| Benchmark data | [gemma4-vs-qwen35-dgx-spark](https://github.com/nabe2030/gemma4-vs-qwen35-dgx-spark) |

## Models

| Model | Quantization | GGUF Size | Source |
|---|---|---|---|
| Gemma 4 26B-A4B-it | Q4_K_M | 15.6 GiB | [ggml-org](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) |
| Qwen 3.5-35B-A3B | Q4_K_M | 19.7 GiB | [ggml-org](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) |

Both tested at Q4_K_M for a fair comparison on the same hardware.

## Results Summary

### llama-bench (Raw Inference Speed)

```bash
llama-bench -m <model.gguf> -ngl 99 -fa 1 -mmp 0 -p 2048 -n 32 -ub 2048
```

![llama-bench速度比較](https://raw.githubusercontent.com/nabe2030/gemma4-vs-qwen35-evo-x2/main/charts/llama_bench_evox2.png)

| Model | Size | pp2048 (tok/s) | tg32 (tok/s) |
|---|---|---|---|
| **Gemma 4 26B-A4B Q4_K_M** | 15.6 GiB | **1,348** | 65.4 |
| Qwen 3.5-35B-A3B Q4_K_M | 19.7 GiB | 730 | **72.9** |
| Gemma 4 26B-A4B F16 | 47.0 GiB | 1,299 | 23.7 |
| Qwen 3.5-35B-A3B Q6_K | 26.6 GiB | 624 | 61.5 |

![GDNの影響](https://raw.githubusercontent.com/nabe2030/gemma4-vs-qwen35-evo-x2/main/charts/gdn_impact_pp.png)

Gemma 4 has **1.8x faster prompt processing** than Qwen 3.5 — because it does not use GDN, the Vulkan backend runs at full efficiency. Qwen 3.5 has a slight edge in token generation (72.9 vs 65.4 t/s) due to smaller active parameters (3B vs 3.8B).

#### EVO-X2 vs DGX Spark

![EVO-X2 vs DGX Spark](https://raw.githubusercontent.com/nabe2030/gemma4-vs-qwen35-evo-x2/main/charts/tg_evox2_vs_dgx.png)

| Model | Quant | DGX Spark tg32 | EVO-X2 tg32 |
|---|---|---|---|
| Qwen 3.5 | MXFP4 / Q4_K_M | 58.0 | **72.9** |
| Gemma 4 | F16 / Q4_K_M | 26.5 | **65.4** |

EVO-X2's Vulkan backend delivers faster token generation than DGX Spark's CUDA despite lower memory bandwidth (256 vs 273 GB/s). The RADV Vulkan driver's recent optimizations (Wave32 FA, graphics queue usage) are the likely cause.

### JCommonsenseQA (Japanese Common Sense Reasoning)

[JCommonsenseQA v1.1](https://huggingface.co/datasets/leemeng/jcommonsenseqa-v1.1) — 1,119 five-choice questions, 3-shot prompting.

![JCQ正解率](https://raw.githubusercontent.com/nabe2030/gemma4-vs-qwen35-evo-x2/main/charts/jcq_accuracy_evox2.png)

| Model | Quant | Think | Accuracy | Latency/q | tok/s |
|---|---|---|---|---|---|
| **Gemma 4** | Q4_K_M | OFF | **96.16%** | 0.42s | 95.6 |
| **Gemma 4** | Q4_K_M | ON | **95.98%** | 7.43s | 60.0 |
| Qwen 3.5 | Q4_K_M | OFF | 94.64% | 0.52s | 103.0 |
| Qwen 3.5 | Q4_K_M | ON | 83.91% | 12.01s | 72.0 |

#### Key Findings

**Gemma 4's Thinking mode barely degrades accuracy (-0.18pt).** This is dramatically different from all other tested combinations:

![Thinking効果](https://raw.githubusercontent.com/nabe2030/gemma4-vs-qwen35-evo-x2/main/charts/thinking_effect_evox2.png)

| Model | Quant | Environment | nothink | think | Degradation |
|---|---|---|---|---|---|
| **Gemma 4 Q4_K_M** | **llama.cpp b8672** | **EVO-X2 Vulkan** | **96.16%** | **95.98%** | **-0.18pt** |
| Gemma 4 Q4_K_M | llama.cpp b8672 | DGX Spark CUDA | — | 95.80% | — |
| Gemma 4 Q4_K_M | Ollama | DGX Spark (ref) | 96.4% | 87.2% | -9.2pt |
| Qwen 3.5 MXFP4 | llama.cpp b8665 | DGX Spark CUDA | 96.16% | 89.28% | -6.88pt |
| Qwen 3.5 Q4_K_M | llama.cpp b8672 | EVO-X2 Vulkan | 94.64% | 83.91% | -10.73pt |

**Gemma 4 also shows stronger quantization resilience.** DGX Spark F16 scored 96.51%, and Q4_K_M scores 96.16% — only 0.35pt loss. Qwen 3.5 drops 1.52pt from MXFP4 (96.16%) to Q4_K_M (94.64%).

### VLM Multimodal Benchmark

5 tech exhibition photos + 3 PPE (safety equipment) detection images.

![VLMレイテンシ](https://raw.githubusercontent.com/nabe2030/gemma4-vs-qwen35-evo-x2/main/charts/vlm_latency_evox2.png)

| Task | Gemma 4 Q4_K_M | Qwen 3.5 Q4_K_M |
|---|---|---|
| Caption (5 images) | **10.29s** | 15.56s |
| JSON extraction (5 images) | **3.70s** | 5.54s |
| JSON parse rate | **100%** | **100%** |
| PPE detection (3 images) | **5.46s** | 6.00s |
| PPE parse rate | **100%** | **100%** |

Gemma 4 is faster across all VLM tasks — 1.5x for captions, 1.5x for JSON extraction. Both models achieve 100% JSON parse rate.

## Known Issue: Gemma 4 Thinking Bug (F16 GGUF)

The `<unused49>` token flood bug reported in the [DGX Spark benchmark](https://github.com/nabe2030/gemma4-vs-qwen35-dgx-spark) was found to be **F16 GGUF-specific**. Quantized GGUFs (Q4_K_M) work correctly on both CUDA and Vulkan.

See: [ggml-org/llama.cpp Discussion #21338](https://github.com/ggml-org/llama.cpp/discussions/21338)

## Build Instructions (EVO-X2 / Strix Halo)

```bash
sudo apt install -y libvulkan-dev

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout b8672  # or latest

cmake -B build-vulkan \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-vulkan --config Release -j$(nproc)
```

## Running Benchmarks

### llama-bench

```bash
./build-vulkan/bin/llama-bench \
  -m <model.gguf> \
  -ngl 99 -fa 1 -mmp 0 \
  -p 2048 -n 32 -ub 2048
```

### JCommonsenseQA

```bash
# Start server (Thinking OFF)
./build-vulkan/bin/llama-server \
  -m <model.gguf> \
  --mmproj <mmproj.gguf> \
  -ngl 99 --jinja --reasoning off \
  --port 8080 --temp 1.0 --top-p 0.95 --top-k 64

# Run benchmark (requires: pip install datasets httpx)
python3 jcq_bench.py \
  --model <model-name> \
  --output results/<model-name>.json
```

### VLM Multimodal

```bash
python3 vlm_bench.py \
  --model <model-name> \
  --image-dir ~/vlm-test-images \
  --output results/<model-name>_vlm.json
```

## Conclusion

On AMD Strix Halo with Vulkan, **Gemma 4 has a clear advantage over Qwen 3.5** — the opposite of what we see on NVIDIA DGX Spark.

| Metric | Winner | Why |
|---|---|---|
| pp speed | Gemma 4 (1.8x) | No GDN → full Vulkan performance |
| tg speed | Qwen 3.5 (1.1x) | Smaller active params (3B vs 3.8B) |
| JCQ accuracy (nothink) | Gemma 4 (+1.5pt) | Better quantization resilience |
| JCQ accuracy (think) | Gemma 4 (+12pt) | Near-zero Thinking degradation |
| VLM speed | Gemma 4 (1.5x) | No GDN bottleneck in image prefill |
| Model size | Gemma 4 (4 GiB smaller) | 15.6 vs 19.7 GiB |
| License | Gemma 4 | Apache 2.0 vs restrictive |

For Strix Halo users running llama.cpp with Vulkan, **Gemma 4 26B-A4B Q4_K_M is the recommended model** — it's faster, more accurate, smaller, and has a permissive license.

## Scripts

| File | Description |
|---|---|
| `jcq_bench.py` | JCommonsenseQA benchmark via OpenAI-compatible API |
| `vlm_bench.py` | VLM multimodal benchmark (caption, JSON extraction, PPE detection) |
| `results/` | Raw benchmark result JSONs |

## References

- [DGX Spark benchmark (companion repo)](https://github.com/nabe2030/gemma4-vs-qwen35-dgx-spark)
- [Gemma 4 DGX Spark Benchmark (DevelopersIO, Ollama)](https://dev.classmethod.jp/articles/dgx-spark-gemma4-benchmark/)
- [llama.cpp Vulkan Performance Discussion](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [GDN Issue on gfx1151](https://github.com/ggml-org/llama.cpp/issues/20354)
- [Strix Halo Wiki: llama.cpp with ROCm](https://strixhalo.wiki/AI/llamacpp-with-ROCm)
- [Phoronix: Ubuntu 26.04 Strix Halo Performance](https://www.phoronix.com/review/ubuntu-2604-strix-halo/4)
- [Known-Good Strix Halo ROCm Stack](https://github.com/ggml-org/llama.cpp/discussions/20856)
- [JCommonsenseQA v1.1](https://huggingface.co/datasets/leemeng/jcommonsenseqa-v1.1)
- [Google Blog: Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)

## Blog Posts

- [Qiita (Japanese): DGX Spark benchmark](https://qiita.com/nabe2030/items/fc3db819470edcca5aee)
- Qiita/Zenn (Japanese): EVO-X2 benchmark — coming soon

## License

Scripts: MIT License
Benchmark data: See individual dataset licenses.
