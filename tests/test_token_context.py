from types import SimpleNamespace

import pandas as pd
import torch

from kvstudy.config import Config, ContextConfig
from kvstudy.token_context.categories import lexical_categories
from kvstudy.token_context.block_attention import recent_page_indices, select_sparse_pages
from kvstudy.token_context.experiment import _distribution_metrics, retained_indices
from kvstudy.token_context.kv_cache import cache_token_count, prune_legacy_cache
from kvstudy.token_context.profile import theoretical_attention_cost
from kvstudy.token_context.report import summarize_context
from kvstudy.token_context.router import cross_validated_type_scores
from kvstudy.token_context.router_exploration import _document_folds
from kvstudy.token_context.sink_cached_experiment import fixed_remote_indices
from kvstudy.token_context.full_kv_distribution import FullKVDistributionRecorder
from kvstudy.token_context.top1_severity import _severity


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


def test_theoretical_attention_cost_rounds_to_small_blocks():
    result = theoretical_attention_cost(
        context_length=16384,
        recent_tokens=2000,
        long_fraction=0.25,
        sparse_remote_tokens=1000,
        block_size=128,
    )
    assert result["rounded_recent_tokens"] == 2048
    assert result["rounded_sparse_remote_tokens"] == 1024
    assert result["v1_mean_kv_reads"] == 5632
    assert result["v2_mean_kv_reads"] == 2304
    assert result["v2_attention_upper_bound_speedup"] > result[
        "v1_attention_upper_bound_speedup"
    ]


def test_sparse_page_selection_always_includes_sink_and_recent():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    landmarks = torch.zeros(8, 1, 2)
    landmarks[3, 0, 0] = 10
    landmarks[4, 0, 1] = 8
    recent = recent_page_indices(8 * 128, 2 * 128, 128)
    selected = select_sparse_pages(query, landmarks, recent, remote_pages=2, sink_pages=1)
    assert selected.tolist() == [0, 3, 4, 6, 7]


def test_router_exploration_folds_whole_documents():
    frame = pd.DataFrame({"document": ["b", "a", "b", "c", "a", "d"]})
    folds = _document_folds(frame)
    assert folds.tolist() == [1, 0, 1, 2, 0, 3]


def test_sink_controls_exclude_prefix_and_are_deterministic():
    random_a = fixed_remote_indices("random_remote", 8, 1000, 17, "cpu")
    random_b = fixed_remote_indices("random_remote", 8, 1000, 17, "cpu")
    strided = fixed_remote_indices("strided_remote", 8, 1000, 17, "cpu")
    assert torch.equal(random_a, random_b)
    assert random_a.min() >= 128 and strided.min() >= 128
    assert len(random_a.unique()) == len(strided.unique()) == 8


def test_full_kv_distribution_block_mass_sums_to_one():
    module = SimpleNamespace(layer_idx=0)
    recorder = FullKVDistributionRecorder([0], 4, "doc", 10)
    query = torch.randn(1, 2, 1, 4)
    key = torch.randn(1, 1, 10, 4)
    value = torch.randn_like(key)
    recorder(module, query, key, value, scaling=0.5)
    assert len(recorder.rows) == 3
    assert abs(sum(row["attention_mass_mean"] for row in recorder.rows) - 1) < 1e-6


def test_top1_severity_prioritizes_correctness_and_surface_changes():
    harmful = pd.Series({
        "top1_changed": True,
        "full_correct": True,
        "compact_correct": False,
        "full_top1_token": " surgeon",
        "compact_top1_token": " physician",
        "top1_embedding_cosine": 0.95,
    })
    surface = pd.Series({
        "top1_changed": True,
        "full_correct": False,
        "compact_correct": False,
        "full_top1_token": "Hello",
        "compact_top1_token": " hello ",
        "top1_embedding_cosine": 0.1,
    })
    assert _severity(harmful) == "definite_harm_full_correct_lost"
    assert _severity(surface) == "surface_or_punctuation"
