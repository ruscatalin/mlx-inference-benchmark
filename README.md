# oMLX benchmark

Measures how fast and how good your local models are when served by oMLX on a Mac.
For every model and setting you choose, it measures speed, energy cost and answer quality. It then
recommends what to run.

## What it measures

| Area | How |
|---|---|
| **Speed** | Total tokens/s and tokens/s per request at several concurrency levels (C), plus time to first token and p95 latency. |
| **Energy cost** | Tokens per joule from the Mac's power reading, turned into a price per million tokens. |
| **Quality** | One score made of two tests, weighted 50/50: **IFBench** (follows instructions, "IF loose") and **MMLU-Pro** (reasoning questions). |
| **Prefix cache** | How much faster a prompt starts when its beginning is already cached (only in runs that include a cache stage). |

Requests that the server rejects in a quality test (for example oMLX's memory guard at high concurrency)
are not counted as wrong answers. They get one more try at the end of that test, one at a time, and only
the ones that still fail are reported as errors. Retried requests are left out of the speed figures.

The quality score is shown with a 95% confidence interval. "Best quality" is the top score. Other
configs still count as equally good unless a paired comparison shows they are clearly worse.

## Quick start

Install the Python packages once (Python 3.11 or newer). The benchmark never installs anything itself;
if a package is missing it stops and tells you what to install.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You also need, installed separately (the benchmark never installs them):

- **oMLX** (required): `brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx`
- **macmon** (optional, for temperature control and energy readings): `brew install macmon`

Then run:

```bash
python3 -m bench run --dry-run           # show the plan and time estimate, run nothing
python3 -m bench run                     # run the default benchmark
python3 -m bench run --preset exhaustive # quality + speed for the baseline, kv8 and kv4 profiles
```

- The models in `models.yaml` are large (tens of GB each). Edit that file to test your own models. Weights go to
  `model_root` (default `~/models/omlx-bench`).
- Data is not software: the first IFBench run also downloads the IFBench test files, MMLU-Pro questions and NLTK
  language data, and the first run of each model downloads its weights.
- Quit the oMLX app first. The benchmark starts its own `omlx serve` and stops if the app's server is running.
- The benchmark never writes to `~/.omlx/`. Each test gets its own settings folder under `~/.omlx-bench/base/`.

## Commands

| Command | What it does |
|---|---|
| `python3 -m bench run` | Start a new benchmark. Prints the plan, then asks `Proceed? [y/N]` (skip with `--yes`). |
| `python3 -m bench resume runs/<run>` | Continue an interrupted run where it stopped. |
| `python3 -m bench report runs/<run>` | Rebuild the report and graphs, optionally with a new price or eligibility ratio. |
| `python3 -m bench rescore runs/<run>` | Score the saved answers again (IFBench and MMLU-Pro). |
| `python3 -m bench dashboard` | Rebuild `runs/index.html`, a page that compares all runs. |
| `python3 -m bench smoke` | Start each model/profile briefly to check the server accepts it. |

## Common options for `run`

| Option | Default | Meaning |
|---|---|---|
| `--preset` | `default` | `default` (quality stage, then speed stage, on the baseline profile), `baseline_quality` (IFBench and MMLU-Pro only, at concurrency 8), `baseline_speed` (speed stage only), `exhaustive` (quality and speed for the baseline, kv8 and kv4 profiles). |
| `--models a,b` | all in `models.yaml` | Which models to test. |
| `--profiles p,q` | `baseline` | Which server settings to test. |
| `--levels 1,2,4,8` | `1,2,3,4,6,8` | Concurrency levels for the speed stage. |
| `--trials` | 3 | Repeats per speed point. More trials give tighter error bars. |
| `--samples` | 10 | Answers per question in the quality stage. |
| `--quality-c` | 1 (8 in `baseline_quality`) | Concurrency for the quality stage. Keep it low: at high values oMLX's memory guard can reject requests. Quality scores do not depend on it. |
| `--quality-prompts N` | all | Use only N IFBench prompts (a big time saver). |
| `--no-quality` | off | Speed only. |
| `--mmlu-questions N` | 300 | MMLU-Pro questions per config. |
| `--mmlu-categories` | math, physics, engineering, computer science | Which MMLU-Pro subjects to draw from. |
| `--no-mmlu` | off | Quality from IFBench only. |
| `--no-thermal` | off | Skip temperature control (uses a fixed cooldown instead). |
| `--price`, `--currency`, `--region` | from `pricing.yaml` | Electricity price for the cost column. |
| `--name` | none | Label added to the run folder name. |
| `--dry-run` | off | Print the plan and stop. |

Run `python3 -m bench run --help` for the full list.

**Time.** Quality is the slow part. Its requests are the IFBench prompts (~300) plus the MMLU-Pro questions, each times `--samples`,
for every model/profile. To cut it down, use `--samples 3`, `--quality-prompts 100` or `--mmlu-questions 100`.
Use `--dry-run` to see the estimate first. The estimate is a fixed formula (server start, thermal wait, plus
requests × max tokens ÷ 40 tok/s). It does not learn from earlier runs. The live **Overall** ETA does correct
itself as cells finish.

## Progress screen

Three lines stay on screen while a run is going:

- **Overall**: progress across every planned cell, with elapsed time and ETA.
- **Thermal**: GPU temperature, the target band, and what the thermal control is doing right now.
- **Current cell**: model, profile, concurrency, live tokens/s and the running quality percentage.

Lines are cut to the window width. If you resize the window while a run is going, a few stray lines may be
left on screen. This is cosmetic and does not affect results.

## Temperature control

Results drift when the Mac gets hot. Before a run, the benchmark heats the GPU to its steady-state
temperature and picks a band around it. It waits for the GPU to be inside that band before each test.
This needs [macmon](https://github.com/vladkens/macmon). Without it, a fixed cooldown is used.

## Configuration

- **`requirements.txt`**: every Python package the benchmark needs.
- **`models.yaml`**: the models (Hugging Face repo, plot colour, optional oMLX settings), the profiles (server and
  per-model settings to compare, such as KV-cache quantization) and the presets.
- **`pricing.yaml`**: electricity price per region. The values are placeholders, so set your own tariff.
  Prices that come from this file are marked "(est.)" in the report.

## Output

Each run is saved in `runs/<run_id>/`. The folder is named by date and time, like `30-09-2026_1051` (day-month-year, then hour and minute). Add `--name kv-test` for a label: `30-09-2026_1051_kv-test`. If the name is already taken, `-2`, `-3` and so on is added.

| File | Content |
|---|---|
| `summary.txt` | The full report (speed, quality, trade-off, verdict). |
| `graphs.png` | Speed, latency, cost, energy, temperature and quality charts. |
| `tradeoff.png` | Quality vs throughput, and throughput vs per-request speed. |
| `run.json`, `cells/`, `responses.jsonl` | Raw data: the plan, every test result and every saved answer. |
| `logs/` | Server logs for each test. |

`runs/index.html` compares all runs in one interactive page. Tick one or more runs on the left. Its
**Recommendations** tab gives one set of picks per run and, when several runs are ticked, one combined set
over all of them (speed and energy averaged, quality weighted towards runs with tighter confidence intervals).
Every model/profile pair has its own colour in the graphs and in the dashboard.

## Project layout

```
bench/
  cli.py        commands and options          sweep.py      runs the tests, draws the progress screen
  plan.py       turns options into tests      thermal.py    temperature band and gating
  omlx.py       starts and checks the server  quality.py    IFBench and MMLU-Pro loading and scoring
  workload.py   sends requests                stats.py      confidence intervals, recommendations
  report.py     console report                plots.py      PNG graphs
  dashboard.py  runs/index.html               runstore.py   saves and loads runs
  config.py     models.yaml                   pricing.py    price per million tokens
  bootstrap.py  checks the packages are installed
tests/fixtures/run_min/   small sample run for checking reports and the dashboard
```

## Checking changes

The benchmark itself takes hours, so use the quick checks instead:

```bash
python3 -m bench run --dry-run
python3 -m bench report tests/fixtures/run_min --no-open
python3 -m bench dashboard --root tests/fixtures --no-open
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
