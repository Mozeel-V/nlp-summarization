"""SAMSum loading and preprocessing.

The selection unit is one utterance, meaning one speaker turn, not a sentence.
The speaker prefix stays attached to the turn.
"""

from __future__ import annotations

from typing import TypedDict

from datasets import DatasetDict, load_dataset

from config import DATASET_NAME, SPLIT_NAMES


class SummaryInput(TypedDict):
    """The fields needed to encode one example."""

    source: str
    summary: str


class Record(SummaryInput):
    """One dialogue text and its reference summary."""

    id: str


class Document(Record):
    """One dialogue split into speaker turns."""

    turns: list[str]


def split_turns(*, dialogue: str) -> list[str]:
    """Split a dialogue into non-empty turns, keeping speaker prefixes."""
    return [line.strip() for line in dialogue.splitlines() if line.strip()]


def load_docs(*, dataset_name: str = DATASET_NAME) -> dict[str, list[Document]]:
    """Load SAMSum and return one list of documents per split."""
    raw = load_dataset(dataset_name)
    if not isinstance(raw, DatasetDict):
        raise TypeError("Expected load_dataset to return a DatasetDict.")
    docs: dict[str, list[Document]] = {}

    for split in SPLIT_NAMES:
        docs[split] = []
        for example in raw[split].to_list():
            turns = split_turns(dialogue=example["dialogue"])
            docs[split].append(
                {
                    "id": example["id"],
                    "source": "\n".join(turns),
                    "summary": example["summary"].strip(),
                    "turns": turns,
                }
            )
    return docs
