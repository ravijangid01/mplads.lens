"""Detector registry."""
from . import agency, cost, delay, duplicate, stalled

REGISTRY = {m.NAME: m.run for m in (cost, duplicate, stalled, agency, delay)}
NAMES = list(REGISTRY)
