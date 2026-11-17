"""format-sumstats: gsMap and COJO outputs, odds-ratio input, P underflow and rsID conversion."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from gsMap.config import FormatSumstatsConfig
from gsMap.format_sumstats import gwas_format

ALLELES = [("A", "C"), ("A", "G"), ("C", "T"), ("G", "T"), ("T", "C"), ("G", "A")]


def _gwas(n_snps=6, **columns):
    frame = pd.DataFrame(
        {
            "SNP": [f"rs{i}" for i in range(n_snps)],
            "A1": [ALLELES[i % len(ALLELES)][0] for i in range(n_snps)],
            "A2": [ALLELES[i % len(ALLELES)][1] for i in range(n_snps)],
            "N": np.full(n_snps, 10_000),
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def _run(tmp_path, frame, **options):
    source = tmp_path / "input.txt"
    frame.to_csv(source, sep="\t", index=False, na_rep="NA")
    config = FormatSumstatsConfig(sumstats=str(source), out=str(tmp_path / "out"), **options)
    gwas_format(config)
    suffix = ".cojo.gz" if options.get("format") == "COJO" else ".sumstats.gz"
    return pd.read_csv(tmp_path / f"out{suffix}", sep="\t")


def test_gsmap_output_signs_z_from_p_and_beta(tmp_path):
    p = np.array([0.5, 1e-3, 1e-10, 0.2, 0.9, 1e-30])
    beta = np.array([0.1, -0.2, 0.3, -0.05, 0.01, -0.4])
    result = _run(tmp_path, _gwas(BETA=beta, SE=np.full(6, 0.05), P=p))
    expected = np.round(np.sqrt(chi2.isf(p, 1)) * np.sign(beta), 3)
    np.testing.assert_allclose(result["Z"], expected, atol=1e-9)
    assert result.columns.tolist() == ["SNP", "A1", "A2", "Z", "N"]


def test_odds_ratio_se_uses_the_delta_method(tmp_path):
    odds = np.array([1.2, 0.8, 1.05, 0.95, 1.5, 0.7])
    se_or = np.array([0.06, 0.04, 0.02, 0.03, 0.1, 0.05])
    result = _run(tmp_path, _gwas(OR=odds, SE_OR=se_or), format="gsMap")
    expected = np.round(np.log(odds) / (se_or / odds), 3)
    np.testing.assert_allclose(result["Z"], expected, atol=1e-9)


def test_odds_ratio_alone_signs_p_values(tmp_path):
    odds = np.array([1.2, 0.8, 1.05, 0.95, 1.5, 0.7])
    p = np.full(6, 1e-4)
    result = _run(tmp_path, _gwas(OR=odds, P=p))
    assert (np.sign(result["Z"]) == np.sign(np.log(odds))).all()


def test_p_values_without_a_direction_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="no direction"):
        _run(tmp_path, _gwas(P=np.full(6, 0.01)))


def test_underflowed_p_values_keep_the_strongest_snps(tmp_path):
    p = np.array([0.0, 1e-5, 0.0, 0.3, 0.5, 0.7])
    beta = np.array([2.0, 0.1, -3.0, 0.2, -0.1, 0.05])
    se = np.full(6, 0.05)
    result = _run(tmp_path, _gwas(BETA=beta, SE=se, P=p)).set_index("SNP")
    assert result.loc["rs0", "Z"] == pytest.approx(40.0)
    assert result.loc["rs2", "Z"] == pytest.approx(-60.0)
    assert len(result) == 6


def test_case_control_sample_sizes_are_summed(tmp_path):
    frame = _gwas(BETA=np.full(6, 0.1), SE=np.full(6, 0.05), P=np.full(6, 0.01)).drop(columns="N")
    frame["N_CAS"] = 4_000
    frame["N_CON"] = 6_000
    result = _run(tmp_path, frame)
    assert (result["N"] == 10_000).all()


def test_missing_values_in_unused_columns_keep_the_snp(tmp_path):
    frame = _gwas(BETA=np.full(6, 0.1), SE=np.full(6, 0.05), P=np.full(6, 0.01))
    frame["DIRECTION"] = ["+", None, "-", None, "+", "+"]
    frame.loc[3, "P"] = np.nan
    result = _run(tmp_path, frame)
    assert result["SNP"].tolist() == ["rs0", "rs1", "rs2", "rs4", "rs5"]


def test_cojo_output_keeps_full_precision(tmp_path):
    frame = _gwas(
        BETA=np.array([0.0004, -0.0002, 0.1, 0.2, -0.3, 0.05]),
        SE=np.full(6, 0.0001),
        P=np.array([1e-8, 2.5e-5, 0.01, 0.2, 0.3, 0.04]),
        FRQ=np.full(6, 0.3),
    )
    result = _run(tmp_path, frame, format="COJO")
    assert result["P"].iloc[0] == pytest.approx(1e-8)
    assert result["BETA"].iloc[0] == pytest.approx(0.0004)


@pytest.mark.parametrize(
    ("gwas_prefix", "dbsnp_prefix"), [("", ""), ("chr", ""), ("chr", "chr"), ("", "chr")]
)
def test_positions_convert_to_rsids_with_any_chromosome_prefix(
    tmp_path, gwas_prefix, dbsnp_prefix
):
    frame = pd.DataFrame(
        {
            "Chr": [f"{gwas_prefix}1", f"{gwas_prefix}1", f"{gwas_prefix}X"],
            "Pos": [100, 200, 300],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "T"],
            "BETA": [0.1, -0.2, 0.3],
            "SE": [0.05, 0.05, 0.05],
            "P": [0.01, 0.02, 0.03],
            "N": [10_000] * 3,
        }
    )
    dbsnp = tmp_path / "dbsnp.tsv"
    dbsnp.write_text(
        "chr\tpos\tref\talt\tdbsnp\n"
        f"{dbsnp_prefix}1\t100\tA\tG\trs100\n"
        # Two variants at one position: the alleles decide.
        f"{dbsnp_prefix}1\t200\tC\tA\trs200a\n"
        f"{dbsnp_prefix}1\t200\tC\tT,G\trs200b\n"
        f"{dbsnp_prefix}X\t300\tG\tT\trs300\n"
    )
    result = _run(tmp_path, frame, dbsnp=str(dbsnp))
    assert sorted(result["SNP"]) == ["rs100", "rs200b", "rs300"]
    assert math.isclose(
        result.set_index("SNP").loc["rs200b", "Z"],
        -round(math.sqrt(chi2.isf(0.02, 1)), 3),
        abs_tol=1e-9,
    )
