# Ollama Mac benchmark

Test every local Ollama model on your Mac with one script, then turn the results into a plain-English report with one prompt. My results below were measured on a Mac mini M6 with 32 GB of unified memory.

## 1. Run the benchmark

Keep Ollama running, then:

```bash
python3 bench.py
```

- Python 3, no extra packages, macOS.
- Tests every downloaded text model on six fixed tasks: a quick question, a word problem, writing code (checked against 8 test cases), following strict rules, finding one access code in a 140-entry log, and a 600-word essay.
- Each scored task runs twice. Temperature 0 and a fixed seed, so repeat runs match.
- Expect one to two minutes per model, longer for large ones.
- Results land in `run-YYYYMMDD-HHMMSS/results.json` next to the script. It talks only to your local Ollama server.

Optional settings (environment variables):

| Variable | Default | What it does |
|---|---|---|
| `BENCH_REPS` | 2 | Repeats per task |
| `BENCH_CTX` | 8192 | Context window |
| `BENCH_COOLDOWN` | 30 | Seconds of rest between models |
| `BENCH_MODELS` | all | Comma list of name pieces to test |
| `BENCH_THINK` | off | Set to 1 to leave thinking mode on |
| `BENCH_HOST` | `http://127.0.0.1:11434` | Ollama address |

For chip temperatures, install macmon first (`brew install macmon`).

## 2. Make the report

Open Claude, paste the contents of `report-prompt.txt`, and attach your `results.json`. Claude builds a self-contained `ollama_benchmark_report.html`: a right-answers grid, a time grid, speed, long-document wait and memory charts, and one card per model with a verdict.

`example-report.html` is the report from my run (Mac mini M6, 32 GB, 8 models).

## My results

- qwen2.5:7b, llama3.1:8b and gemma3:12b got all 8 scored answers right. qwen2.5:7b was the fastest of the three (25.3 words/s).
- llama3.2:1b was the fastest overall (84.3 words/s) but scored 3 of 8.
- qwen3.8:27b-mlx scored 7 of 8, waited 14.3 s before its first word on the long log, used 18.5 GB and moved 527 MB to swap.

Words per second is an estimate: tokens per second times 0.75.
