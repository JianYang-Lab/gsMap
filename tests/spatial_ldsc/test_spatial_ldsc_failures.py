"""Failure-semantics tests for the multi-trait spatial LDSC entry point."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gsMap.spatial_ldsc import spatial_ldsc_jax


def test_failed_trait_does_not_turn_the_multi_trait_run_into_success(tmp_path, monkeypatch):
    processed = []
    aborted = []

    class FakeWriter:
        closed = False

        def abort(self):
            self.closed = True
            aborted.append(True)

    class FakeProcessor:
        def __init__(self, **kwargs):
            del kwargs

        def setup_trait(self, trait_name, sumstats_file):
            del sumstats_file
            self.result_writer = FakeWriter()
            if trait_name == "bad":
                raise RuntimeError("synthetic setup failure")

        def process_all_chunks(self):
            processed.append("good")
            self.result_writer.closed = True

    config = SimpleNamespace(
        sumstats_config_dict={"bad": "bad.sumstats", "good": "good.sumstats"},
        ldsc_save_dir=tmp_path / "spatial_ldsc",
        ldscore_h5ad_path=tmp_path / "direct.h5ad",
        sample_filter=None,
        cell_indices_range=None,
        project_name="test",
    )
    monkeypatch.setattr(spatial_ldsc_jax.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(
        spatial_ldsc_jax,
        "load_common_resources_direct_ldscore",
        lambda _config: (object(), object(), object()),
    )
    monkeypatch.setattr(spatial_ldsc_jax, "SpatialLDSCEngine", FakeProcessor)
    monkeypatch.setattr(
        spatial_ldsc_jax,
        "generate_expected_output_filename",
        lambda _config, _trait, **_kwargs: None,
    )
    monkeypatch.setattr(
        spatial_ldsc_jax,
        "stage_logging",
        lambda *args, **kwargs: nullcontext(),
    )

    with pytest.raises(RuntimeError, match=r"failed for 1/2 traits: bad"):
        spatial_ldsc_jax.run_spatial_ldsc_jax.__wrapped__(config)

    assert processed == ["good"]
    assert aborted == [True]
