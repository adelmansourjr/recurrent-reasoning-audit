# Recurrent Qwen and Symbolic Verification

I trained local Qwen3-1.7B LoRA adapters and checked the typed programs they produced with an exact arithmetic interpreter. This folder publishes the original implementation alongside the archived experiment audit in the parent directory.

[Recurrent mathematical model](src/latent_loop/recurrent_math.py) · [Typed arithmetic interpreter](src/latent_loop/mathematics.py) · [Mathematical adapter pipeline](research/math_adapter.py) · [Shared adapter training](research/semantic_adapter.py)

The source was extracted from the local loop-transformer project. In this public copy, the model default reads the QWEN_MODEL_DIRECTORY environment variable. Set it to an existing MLX checkpoint on Apple Silicon. The source needs the packages in requirements.txt.

```sh
export QWEN_MODEL_DIRECTORY=/path/to/Qwen3-1.7B-4bit
PYTHONPATH=training/src python3 training/research/math_adapter.py --help
```

Run from the repository root. The prepare, train and evaluate stages require the original input datasets and preparation outputs. The scripts document the typed program format and train/test handling. The parent audit gives the measured accuracy limits and exact replay commands. This source publication adds no new training run.
