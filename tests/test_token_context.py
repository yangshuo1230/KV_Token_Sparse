from types import SimpleNamespace

import pandas as pd
import torch

from kvstudy.config import Config, ContextConfig
from kvstudy.token_context.categories import lexical_categories
from kvstudy.token_context.experiment import _distribution_metrics, retained_indices
from kvstudy.token_context.kv_cache import cache_token_count, prune_legacy_cache
from kvstudy.token_context.report import summarize_context
from kvstudy.token_context.router import cross_validated_type_scores


def _token(text: str, pos: str, dep: str = "", punct: bool = False, number: bool = False):
    return SimpleNamespace(
        lower_=text.lower(), pos_=pos, dep_=dep, is_punct=punct, like_num=number
    )


def test_lexical_categories_are_fine_grained():
    assert lexical_categories(_token("Paris", "PROPN")) == ("content", "proper_noun")
    assert lexical_categories(_token("because", "SCONJ")) == (
        "function", "subordinating_conjunction"
    )
    assert lexical_categories(_token("not", "PART", dep="neg")) == ("special", "negation")
    assert lexical_categories(_token("who", "PRON")) == ("special", "question_word")


def test_retained_indices_use_a_fixed_budget():
    recent_only = retained_indices(length=1000, cache_budget=128, sink_size=0)
    sink_recent = retained_indices(length=1000, cache_budget=128, sink_size=4)
    assert recent_only == list(range(872, 1000))
    assert sink_recent[:4] == [0, 1, 2, 3]
    assert sink_recent[4:] == list(range(876, 1000))
    assert len(recent_only) == len(sink_recent) == 128


def test_prune_legacy_cache_retains_sink_and_recent_values():
    values = torch.arange(20).view(1, 1, 10, 2)
    cache = ((values, values + 100),)
    pruned = prune_legacy_cache(cache, cache_budget=4, sink_size=1)
    assert cache_token_count(pruned) == 4
    assert pruned[0][0][0, 0, :, 0].tolist() == [0, 14, 16, 18]
    assert pruned[0][1][0, 0, :, 0].tolist() == [100, 114, 116, 118]


def test_distribution_metrics_use_target_token():
    full = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    compact = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
    metrics = _distribution_metrics(full, compact, torch.tensor([1, 0]))
    assert metrics["delta_ce"][0] > 0
    assert metrics["top1_changed"] == [True, False]


def test_context_summary_pairs_sink_with_recent_control(tmp_path):
    rows = []
    for document, scale in (("a", 1.0), ("b", 2.0)):
        for target_index, label in ((10, "noun"), (11, "verb")):
            for sink_size in (0, 4):
                value = scale + (0.25 if sink_size else 0.0)
                rows.append({
                    "document": document,
                    "target_index": target_index,
                    "cache_budget": 128,
                    "sink_size": sink_size,
                    "coarse_category": "content",
                    "fine_category": label,
                    "pos": label.upper(),
                    "is_first_subtoken": True,
                    "delta_ce": value,
                    "kl_full_to_compact": value / 2,
                    "top1_changed": float(value > 1),
                })
    pd.DataFrame(rows).to_parquet(tmp_path / "context_metrics.parquet", index=False)
    cfg = Config(
        model="unused",
        output_dir=tmp_path,
        corpora={"pg19": 2},
        context=ContextConfig(
            cache_budgets=[128], sink_sizes=[0, 4], eval_tokens=8, bootstrap_samples=50
        ),
    )
    summary, contrasts, sink, pairs = summarize_context(cfg)
    assert summary.exists() and contrasts.exists() and sink.exists() and pairs.exists()
    sink_frame = pd.read_csv(sink)
    selected = sink_frame[
        (sink_frame.label_scheme == "fine_category")
        & (sink_frame.label == "noun")
        & (sink_frame.metric == "delta_ce")
    ]
    assert selected.iloc[0].mean_difference == 0.25


def test_type_lookup_scores_are_out_of_document():
    frame = pd.DataFrame({
        "document": ["a", "a", "b", "b"],
        "query_token_id": [1, 1, 2, 2],
        "coarse_category": ["content", "content", "function", "function"],
    })
    scores = cross_validated_type_scores(frame, "query_token_id", folds=2)
    # Each fold contains an unseen token ID, so it must use the opposite
    # document's prior rather than leaking the held-out labels.
    assert scores.tolist() == [0.0, 0.0, 1.0, 1.0]
