import subprocess
import sys
import time
import os
from pathlib import Path

def run_benchmark():
    """
    Executes a controlled training run of Model B using the PyTorch backend.
    Captures throughput, VRAM, and execution time.
    """
    # Bounded launch benchmark. Increase only after this smoke path passes.
    PYTHON_EXEC = sys.executable
    TRAIN_SCRIPT = "train_model.py"
    CHECKPOINT_DIR = None  # Avoid writing a multi-GiB optimizer checkpoint during smoke tests.
    MODEL_NAME = "dual-resolution-matrix-pool-sparse-1024-20l"

    # Real Asset Paths
    SHARD_DIR = "token_shards_ensemble_32k_mixed"
    TOKENIZER_PATH = "token_shards_ensemble_32k_mixed/tokenizer.json"

    # Benchmark settings
    BATCH_SIZE = 1
    GRAD_ACCUM = 1
    TOTAL_STEPS = 1
    CONTEXT_LENGTH = 128
    PRECISION = "bf16-mixed"
    print("="*60)
    print("STARTING PYTORCH MODEL B BENCHMARK (SPARSE RETRIEVAL)")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch Size: {BATCH_SIZE}, Grad Accum: {GRAD_ACCUM}")
    print(f"Total Steps: {TOTAL_STEPS}")
    print("="*60)

    if CHECKPOINT_DIR is not None:
        Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Build the command
    # Note: We use MINI_LLM_BACKEND=pytorch to ensure the engine selector works
    cmd = [
        PYTHON_EXEC, TRAIN_SCRIPT,
        "--backend", "pytorch",
        "--model", MODEL_NAME,
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum-steps", str(GRAD_ACCUM),
        "--precision", PRECISION,
        "--context-length", str(CONTEXT_LENGTH),
        "--total-steps", str(TOTAL_STEPS),
        "--max-run-steps", str(TOTAL_STEPS),
        "--optimizer", "hybrid-muon",
        "--log-interval", "1",
        "--tokenizer-path", TOKENIZER_PATH,
        "--shard-dir", SHARD_DIR,
    ]
    if CHECKPOINT_DIR is not None:
        cmd.extend(("--checkpoint-dir", CHECKPOINT_DIR))
    environment = os.environ.copy()
    environment["MINI_LLM_BACKEND"] = "pytorch"
    torch_site = Path(".venv-0089b-torch/lib/python3.12/site-packages").resolve()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(torch_site) if not existing_pythonpath
        else str(torch_site) + os.pathsep + existing_pythonpath
    )
    print("Command:", " ".join(cmd))

    start_time = time.perf_counter()

    try:
        # We run the process and capture output
        # Using subprocess.run allows us to wait for completion and check exit code
        process = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            env=environment,
        )

        end_time = time.perf_counter()
        duration = end_time - start_time

        print("\n[SUCCESS] Training run completed.")
        print(f"Total wall clock time: {duration:.2f}s")

        # Parsing the stdout for metrics (assuming standard logger format)
        # We look for lines like: "Supervised throughput: 25000.0 tokens/s"
        output = process.stdout
        print("\n--- LOG OUTPUT ---")
        print(output)
        print("------------------\n")

        # In a real production environment, we would parse the log/CSV
        # to extract supervised_tokens/s and VRAM usage.
        print("Note: For full performance reports, please check the generated CSV logs.")

    except subprocess.CalledProcessError as e:
        print("\n[FAILURE] Training run failed.")
        print(f"Exit Code: {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise e
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    run_benchmark()
