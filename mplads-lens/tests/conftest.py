import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import clean, run, score, synthetic  # noqa: E402
from pipeline.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def works(cfg):
    return clean.clean(synthetic.generate(seed=cfg["seed"], as_of=cfg["as_of"]))


@pytest.fixture(scope="session")
def dets(works, cfg):
    return run.run_all(works, cfg)


def recall_precision(works, dets, planted, name):
    pl = works.set_index("work_id")["_planted_type"]
    truth, hit = set(pl[pl == planted].index), set(dets[dets.detector == name].work_id)
    return len(truth & hit) / len(truth), len(truth & hit) / max(len(hit), 1)
