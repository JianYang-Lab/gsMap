"""Data-stage readiness never depends on a current or future GWAS panel."""
from collections import OrderedDict
from dataclasses import fields

import anndata as ad
import numpy as np
import pytest

from gsMap.config import DataPreparationConfig, FindLatentRepresentationsConfig


def source(tmp_path, embedding=False, expression_layer=False):
    data = ad.AnnData(np.ones((4, 25), dtype=np.float32))
    data.obs['celltype'] = 'author_type'
    data.obsm['spatial'] = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    if embedding:
        data.obsm['cell_emb_pca'] = np.ones((4, 2), dtype=np.float32)
    if expression_layer:
        data.layers['processed'] = data.X - 2
    path = tmp_path / 'source.h5ad'
    data.write_h5ad(path)
    return path


def test_counts_sources_validate_without_future_embeddings_or_gwas(tmp_path):
    path = source(tmp_path)
    config = DataPreparationConfig(workdir=tmp_path, project_name='prepare',
                                   sample_h5ad_dict=OrderedDict([('section_B', str(path))]),
                                   annotation='celltype', use_gpu=False)
    assert not hasattr(config, '_init_sumstats')
    assert not any(field.name.startswith(('sumstats', 'trait')) for field in fields(config))
    assert list(config.sample_h5ad_dict) == ['section_B']
    assert config.latent_representation_cell == 'emb_cell'
    assert isinstance(config.find_latent_config, FindLatentRepresentationsConfig)
    # The second-stage initializer is intentionally deferred until first-stage
    # output files exist, rather than reporting source counts as broken.
    with pytest.raises(ValueError, match='emb_cell'):
        _ = config.latent2gene_config


def test_noncounts_uses_supplied_pca_without_jax_or_gwas(tmp_path, monkeypatch):
    path = source(tmp_path, embedding=True, expression_layer=True)
    monkeypatch.setattr('gsMap.config.latent2gene_config.configure_jax_platform',
                        lambda _: pytest.fail('preparation must not initialize JAX'))
    config = DataPreparationConfig(
        workdir=tmp_path, project_name='noncounts', h5ad_path=[path],
        annotation='celltype', expression_data_type='noncounts',
        data_layer='processed', start_step='latent2gene',
        no_expression_fraction=True, latent_representation_cell='cell_emb_pca',
        use_gpu=False,
    )
    assert config.expression_data_type == 'noncounts'
    assert config.high_quality_neighbor_filter is False
    assert config.latent_representation_cell == 'cell_emb_pca'


@pytest.mark.parametrize(('change', 'message'), [
    ({'latent_representation_cell': None}, 'existing named'),
    ({'latent_representation_cell': 'missing_pca'}, 'missing_pca'),
    ({'start_step': 'find_latent'}, 'skip find_latent'),
    ({'stop_step': 'report'}, 'only find_latent and latent2gene'),
    ({'cell_qc_min_counts': 1}, 'cell_qc_min_counts'),
    ({'homolog_file': '/nonexistent/homolog'}, 'pre-mapped gene axis'),
    ({'data_layer': 'missing_processed'}, 'missing_processed'),
])
def test_noncounts_invalid_sources_are_blocked(tmp_path, change, message):
    path = source(tmp_path, embedding=True)
    params = dict(workdir=tmp_path, project_name='bad', h5ad_path=[path],
                  expression_data_type='noncounts', start_step='latent2gene',
                  no_expression_fraction=True, latent_representation_cell='cell_emb_pca')
    with pytest.raises(ValueError, match=message):
        DataPreparationConfig(**dict(params, **change))


def test_existing_latent_metadata_never_replaces_selected_source(tmp_path):
    path = source(tmp_path)
    latent = tmp_path / 'prepare' / 'find_latent_representations'
    latent.mkdir(parents=True)
    (latent / 'find_latent_metadata.yaml').write_text('outputs:\n  latent_files:\n    WRONG: /not/selected.h5ad\n')
    config = DataPreparationConfig(workdir=tmp_path, project_name='prepare', h5ad_path=[path])
    assert list(config.sample_h5ad_dict.values()) == [path]


def test_counts_graph_parameters_use_actual_first_stage_validator(tmp_path):
    path = source(tmp_path)
    with pytest.raises(ValueError, match='positive graph_radius'):
        DataPreparationConfig(workdir=tmp_path, project_name='bad_graph', h5ad_path=[path],
                              graph_mode='radius')
