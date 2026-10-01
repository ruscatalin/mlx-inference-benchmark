"""IFBench and MMLU-Pro dataset loading and scoring."""
import contextlib
import importlib
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

IFBENCH_GIT = "https://github.com/allenai/IFBench"


def ifbench_data(d):
    d = Path(d).expanduser()
    if not any(d.glob("**/IFBench_test.jsonl")):
        print(f"  cloning {IFBENCH_GIT} → {d}")
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", IFBENCH_GIT, str(d)])
    data = sorted(d.glob("**/IFBench_test.jsonl"))
    if not data:
        raise RuntimeError(f"IFBench_test.jsonl not found under {d}")
    prompts = {}
    for line in data[0].read_text().splitlines():
        if line.strip():
            ex = json.loads(line)
            prompts[str(ex.get("key", ex.get("id")))] = ex["prompt"]
    return data[0], prompts


class IFScorer:
    """Official allenai/IFBench checks, imported from the cloned repo (deps from requirements.txt)."""

    def __init__(self, repo_dir, data_path):
        libs = sorted(Path(repo_dir).expanduser().glob("**/evaluation_lib.py"))
        if not libs:
            raise RuntimeError("evaluation_lib.py not found in IFBench repo")
        sys.path.insert(0, str(libs[0].parent))
        try:
            ev = importlib.import_module("evaluation_lib")
        except ModuleNotFoundError as e:
            raise RuntimeError(f"IFBench needs the Python package '{e.name}': "
                               f"pip install -r requirements.txt") from e
        with contextlib.suppress(Exception):  # NLTK data is best-effort; IFBench reports its own errors
            import nltk
            for pkg, path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                              ("stopwords", "corpora/stopwords"),
                              ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                              ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")):
                try:
                    nltk.data.find(path)
                except LookupError:
                    nltk.download(pkg, quiet=True)
        self.ev = ev
        self.lock = threading.Lock()
        self.inputs = {str(x.key): x for x in ev.read_prompt_list(str(data_path))}

    def score(self, pid, response):
        inp = self.inputs[pid]
        m = {inp.prompt: response}
        with self.lock:
            s = self.ev.test_instruction_following_strict(inp, m)
            lo = self.ev.test_instruction_following_loose(inp, m)
        return {"strict": bool(s.follow_all_instructions), "loose": bool(lo.follow_all_instructions),
                "inst_strict": [bool(x) for x in s.follow_instruction_list],
                "inst_loose": [bool(x) for x in lo.follow_instruction_list]}


# ---- MMLU-Pro (TIGER-Lab/MMLU-Pro, test split) ---------------------------------------------
MMLU_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_FILE = "data/test-00000-of-00001.parquet"
MMLU_CATEGORIES = ("math", "physics", "engineering", "computer science")
MMLU_ALL_CATEGORIES = ("math", "physics", "chemistry", "engineering", "computer science", "biology", "health",
                       "business", "economics", "law", "philosophy", "psychology", "history", "other")
LETTERS = "ABCDEFGHIJ"


def mmlu_data(categories=MMLU_CATEGORIES):
    """qid -> {question, options, answer, category} for the given categories (downloaded once into
    the HuggingFace cache)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(MMLU_REPO, MMLU_FILE, repo_type="dataset")
    cats = set(categories)
    return {f"mmlu/{r['question_id']}": {"question": r["question"], "options": r["options"],
                                        "answer": r["answer"], "category": r["category"]}
            for r in pq.read_table(path).to_pylist() if r["category"] in cats}


def mmlu_prompt(item):
    """Zero-shot chain-of-thought prompt in the official MMLU-Pro format."""
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    return (f"The following is a multiple choice question about {item['category']}. Think step by step "
            f'and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n'
            f"Question: {item['question']}\nOptions:\n{opts}\nAnswer: Let's think step by step.")


_ANSWER_RES = (re.compile(r"answer is\s*\**\s*\(?([A-J])\b"), re.compile(r"[aA]nswer:\s*\**\s*\(?([A-J])\b"))
_LAST_LETTER = re.compile(r"\b([A-J])\b")


def mmlu_extract(text):
    """Official MMLU-Pro extraction order ("answer is (X)", then "Answer: X", then the last lone
    letter), taking the last match so a model that revises its answer is scored on the final one."""
    for rx in _ANSWER_RES:
        m = rx.findall(text)
        if m:
            return m[-1]
    m = _LAST_LETTER.findall(text)
    return m[-1] if m else None


def mmlu_score(item, response):
    pred = mmlu_extract(response)
    return {"correct": pred == item["answer"], "pred": pred}
