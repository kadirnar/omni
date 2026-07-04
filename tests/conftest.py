"""Shared fixtures for the omni test suite.

Everything here is CPU-only and offline: no network, no CUDA, tiny dims.
Peer modules are imported lazily inside fixtures so that test files whose
dependencies are not implemented yet fail in isolation, not at collection
of the whole suite.
"""

from __future__ import annotations

import os

import pytest

# Hard offline guarantees: the suite must never touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: heavy/optional test, only runs with RUN_SLOW=1"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip_slow = pytest.mark.skip(reason="slow test: set RUN_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def test_cfg():
    """Tiny-but-real OmniConfig shared across the suite.

    2 codebooks (S=3 streams), d_model=64, 2 layers, GQA 2q/1kv,
    ByteTokenizer-sized text vocab (320), FakeCodec, batch_size=2.
    """
    from omni.config import load_config

    return load_config(
        "tiny",
        [
            "model.n_codebooks=2",
            "model.d_model=64",
            "model.n_layers=2",
            "model.n_heads=2",
            "model.n_kv_heads=1",
            "model.d_ff=128",
            "model.max_frames=128",
            "model.text_vocab_size=320",
            "data.max_sample_frames=120",
            "data.batch_size=2",
            "data.num_workers=0",
            "train.log_every=1",
            "codec=fake",
            "tokenizer_path=byte",
        ],
    )


@pytest.fixture(scope="session")
def byte_tok():
    from omni.text.tokenizer import ByteTokenizer

    return ByteTokenizer()


@pytest.fixture(scope="session")
def fake_codec(test_cfg):
    from omni.audio.codec import FakeCodec

    return FakeCodec(
        n_codebooks=test_cfg.model.n_codebooks,
        codec_vocab=test_cfg.model.audio_codec_vocab,
    )


@pytest.fixture(scope="session")
def fake_shards(tmp_path_factory: pytest.TempPathFactory, test_cfg):
    """Shard dir with 24 tiny fake samples (SineTTS + FakeCodec + ByteTokenizer)."""
    from omni.data.prepare import prepare_fake

    out = tmp_path_factory.mktemp("fake_shards")
    prepare_fake(out, n_samples=24, cfg=test_cfg, seed=0)
    return out


# ---------------------------------------------------------------------------
# Extensions v2 fixtures (depth transformer / full duplex). Derived from
# test_cfg; dataclasses.replace re-runs ModelConfig.__post_init__, so the
# flat<->use_depth config gate is enforced at fixture build time too.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def depth_cfg(test_cfg):
    """test_cfg with the v2 depth overrides: flat delays + depth transformer.

    depth_d_model=32, 2 depth layers, 2 depth heads; max_delay becomes 1, so
    data.max_sample_frames=120 still fits model.max_frames=128.
    """
    import copy
    import dataclasses

    cfg = copy.deepcopy(test_cfg)
    cfg.model = dataclasses.replace(
        cfg.model,
        audio_delay_mode="flat",
        use_depth=True,
        depth_d_model=32,
        depth_n_layers=2,
        depth_n_heads=2,
    )
    assert cfg.data.max_sample_frames + cfg.model.max_delay <= cfg.model.max_frames
    return cfg


@pytest.fixture(scope="session")
def duplex_cfg(test_cfg):
    """test_cfg with full-duplex streams (S = 1 + 2*n_codebooks, stagger delays)."""
    import copy
    import dataclasses

    cfg = copy.deepcopy(test_cfg)
    cfg.model = dataclasses.replace(cfg.model, duplex=True)
    assert cfg.data.max_sample_frames + cfg.model.max_delay <= cfg.model.max_frames
    return cfg


@pytest.fixture(scope="session")
def duplex_shards(tmp_path_factory: pytest.TempPathFactory, duplex_cfg):
    """Shard dir with 10 tiny full-duplex conversations (SineTTS + FakeCodec)."""
    from omni.audio.codec import FakeCodec
    from omni.data.prepare import prepare_duplex
    from omni.data.synthesize import SineTTS
    from omni.text.tokenizer import ByteTokenizer

    out = tmp_path_factory.mktemp("duplex_shards")
    prepare_duplex(
        out,
        n_conversations=10,
        cfg=duplex_cfg,
        tts=SineTTS(),
        codec=FakeCodec(
            n_codebooks=duplex_cfg.model.n_codebooks,
            codec_vocab=duplex_cfg.model.audio_codec_vocab,
        ),
        tokenizer=ByteTokenizer(),
        seed=0,
    )
    return out
