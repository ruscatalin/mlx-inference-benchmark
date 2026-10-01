# TODO

## Offline mode
Let the benchmark run with no internet once everything is already on disk (IFBench data, MMLU-Pro data,
NLTK data, model weights).

- Load MMLU-Pro from the local Hugging Face cache without contacting the Hub.
- `bench/omlx.py`: skip `snapshot_download` when the model folder already exists.
- The NLTK data check is already in place.
