"""pkit test suite — fast, CPU-only, no network, no model loading.

Covers pkit.measures / axes / cooking / captions / facets / load.
pkit.extraction (torch) is deliberately never imported.

Degenerate-input tests pin the eps-guard behavior (NaN/crash -> documented
sensible values); healthy-input regression tests pin bit-identical behavior
via values hardcoded from the pre-guard code (numpy repr round-trips floats
exactly).
"""
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pkit import axes, captions, cooking, measures, paths  # noqa: E402

assert "torch" not in sys.modules, "pkit import must not pull torch"


# ===================================================================
# measures
# ===================================================================

class TestIpsatize:
    def test_flat_row_maps_to_zeros(self):
        M = np.array([[3.0, 3.0, 3.0, 3.0],
                      [1.0, 2.0, 3.0, 4.0]])
        out = measures.ipsatize(M)
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out[0], np.zeros(4))

    def test_cg_shape_identity(self):
        # ipsatized row == sqrt(k) * unit-normalized mean-centered row
        rng = np.random.default_rng(0)
        M = rng.normal(size=(8, 12))
        out = measures.ipsatize(M)
        k = M.shape[1]
        Mc = M - M.mean(axis=1, keepdims=True)
        shape = np.sqrt(k) * Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
        for i in range(M.shape[0]):
            cos = out[i] @ shape[i] / (np.linalg.norm(out[i])
                                       * np.linalg.norm(shape[i]))
            assert abs(cos - 1.0) < 1e-9

    def test_healthy_regression(self):
        # hardcoded from pre-guard code, rng(42) 5x4 — must be bit-identical
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5, 4))
        out = measures.ipsatize(X)
        np.testing.assert_array_equal(
            out[0], [0.08502965783285328, -1.6531845104768783,
                     0.6612032820476598, 0.9069515705963652])
        np.testing.assert_array_equal(
            out[3], [-0.1866851376823615, 1.2880060962139108,
                     0.3712221132108318, -1.472543071742381])


class TestCosSim:
    def test_degenerate_zero_norm_rows_finite(self):
        # constant matrix: after column-centering every row has zero norm
        S = measures.cos_sim(np.ones((3, 2)))
        assert np.all(np.isfinite(S))
        np.testing.assert_array_equal(S, np.zeros((3, 3)))

    def test_single_flat_row_among_healthy(self):
        # rows r, -r, 0: column means are exactly 0 (r + (-r) + 0), so
        # centering is a no-op and row 2 has exactly zero norm
        rng = np.random.default_rng(1)
        r = rng.normal(size=6)
        X = np.vstack([r, -r, np.zeros(6)])
        S = measures.cos_sim(X)
        assert np.all(np.isfinite(S))
        np.testing.assert_array_equal(S[2], np.zeros(3))
        assert S[0, 1] == pytest.approx(-1.0)  # healthy rows unaffected

    def test_healthy_regression(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5, 4))
        S = measures.cos_sim(X)
        np.testing.assert_array_equal(
            S[0], [0.9999999999999998, -0.4871029696765874,
                   0.9490394357633941, -0.6347652490464127,
                   0.47179368752943096])
        assert S[2, 4] == 0.31492720428664844
        np.testing.assert_array_equal(S, S.T)


class TestZscoreOffdiag:
    def test_degenerate_constant_finite(self):
        Z = measures.zscore_offdiag(np.ones((4, 4)))
        assert np.all(np.isfinite(Z))
        np.testing.assert_array_equal(measures.offdiag(Z), np.zeros(12))

    def test_healthy_regression(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5, 4))
        Z = measures.zscore_offdiag(measures.cos_sim(X))
        assert Z[0, 1] == -0.641551196959484
        assert Z[3, 2] == -0.9571549837838723
        off = measures.offdiag(Z)
        assert abs(off.mean()) < 1e-12
        assert abs(off.std() - 1.0) < 1e-12


class TestWinsorize:
    def test_degenerate_empty_keep_returns_unchanged(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 2))
        out = measures.winsorize(X.copy(), [0, 1])  # every column "massive"
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out, X)

    def test_degenerate_zero_cap_flattens(self):
        X = np.column_stack([np.ones(5), np.arange(5.0)])
        out = measures.winsorize(X.copy(), [1])  # keep column is constant
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out[:, 1], np.zeros(5))

    def test_healthy_regression(self):
        rng = np.random.default_rng(3)
        W = rng.normal(size=(20, 4))
        W[:, 1] *= 10.0
        out = measures.winsorize(W.copy(), [1])
        # column 1 capped to exactly the max std of the other columns
        assert out[:, 1].std() == 1.2321120632564218
        np.testing.assert_array_equal(
            out[0], [2.0409191213851825, -2.4202625714173824,
                     0.41809884672577885, -0.5677696061279298])


class TestOffdiagCorr:
    def test_degenerate_constant_finite(self):
        rng = np.random.default_rng(0)
        B = rng.normal(size=(3, 3))
        assert measures.offdiag_corr(np.ones((3, 3)), B) == 0.0
        assert measures.offdiag_corr(B, np.full((3, 3), 7.0)) == 0.0

    def test_healthy_regression(self):
        rng = np.random.default_rng(42)
        S = measures.cos_sim(rng.normal(size=(5, 4)))
        assert measures.offdiag_corr(S, S**2) == 0.4039866465560742
        rng2 = np.random.default_rng(7)
        assert (measures.offdiag_corr(S, rng2.normal(size=(5, 5)))
                == -0.06060649864512024)
        assert measures.offdiag_corr(S, S) == pytest.approx(1.0)


class TestDistReadouts:
    def test_ev_healthy_regression(self):
        assert measures.ev_from_dist({"1": 0.1, "2": 0.2, "7": 0.4}) \
            == 4.714285714285714

    def test_ev_degenerate_finite(self):
        assert measures.ev_from_dist({}) == 0.0
        # zero mass -> uniform over keys -> mean of keys
        assert measures.ev_from_dist({"1": 0.0, "7": 0.0}) == 4.0

    def test_entropy_healthy_regression(self):
        assert measures.entropy_from_dist({"1": 0.1, "2": 0.2, "7": 0.4}) \
            == 0.9556998911125343

    def test_entropy_degenerate_finite(self):
        assert measures.entropy_from_dist({}) == 0.0
        assert measures.entropy_from_dist({"1": 0.0}) == 0.0
        # point mass -> zero entropy, healthy path
        assert measures.entropy_from_dist({"4": 1.0}) == 0.0


class TestRemovePc1:
    def test_removes_top_eigencomponent(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(6, 8))
        S = np.atleast_2d(np.corrcoef(X))
        A = S.copy()
        np.fill_diagonal(A, 0.0)
        w, v = np.linalg.eigh(A)
        k = np.argmax(np.abs(w))
        out = measures.remove_pc1(S)
        # the removed component's eigenvector now has eigenvalue ~0 ...
        assert np.linalg.norm(out @ v[:, k]) < 1e-9
        # ... and the spectrum equals A's spectrum with that entry zeroed
        expect = w.copy()
        expect[k] = 0.0
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(out)),
                                   np.sort(expect), atol=1e-9)


# ===================================================================
# axes
# ===================================================================

class TestVarimax:
    def test_same_subspace_as_input(self):
        rng = np.random.default_rng(11)
        Phi = rng.normal(size=(20, 4))
        L = axes.varimax(Phi)
        Q, _ = np.linalg.qr(Phi)          # orthonormal basis of col(Phi)
        resid = L - Q @ (Q.T @ L)          # projection distance to col(Phi)
        assert np.linalg.norm(resid) < 1e-8

    def test_recovers_simple_structure(self):
        # perfect simple structure, scrambled by a random rotation
        rng = np.random.default_rng(2)
        Phi0 = np.zeros((12, 3))
        for j in range(3):
            Phi0[4 * j:4 * (j + 1), j] = 0.8
        R0, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        L = axes.varimax(Phi0 @ R0)
        # every variable should load on exactly one recovered factor
        dominance = np.max(np.abs(L), axis=1) / np.linalg.norm(L, axis=1)
        assert np.all(dominance > 0.999)

    def test_normalize_flag_runs(self):
        rng = np.random.default_rng(3)
        Phi = rng.normal(size=(10, 3))
        L = axes.varimax(Phi, normalize=True)
        assert L.shape == Phi.shape
        assert np.all(np.isfinite(L))


class TestParticipationRatio:
    def test_k_equal_eigenvalues(self):
        assert axes.participation_ratio([2.0] * 7) == 7.0
        assert axes.participation_ratio(np.ones(3)) == 3.0

    def test_single_dominant(self):
        assert axes.participation_ratio([1.0]) == 1.0

    def test_degenerate_all_nonpositive_finite(self):
        assert axes.participation_ratio([-1.0, 0.0]) == 0.0
        assert axes.participation_ratio([]) == 0.0


class TestHornK:
    def test_pure_noise_retains_nothing(self):
        rng = np.random.default_rng(123)
        R = rng.normal(size=(200, 8))
        assert axes.horn_k(R, n_perm=20, seed=0) <= 1

    def test_one_real_factor(self):
        rng = np.random.default_rng(9)
        f = rng.normal(size=(300, 1))
        R = f @ np.ones((1, 6)) + 0.5 * rng.normal(size=(300, 6))
        assert axes.horn_k(R, n_perm=20, seed=0) >= 1


class TestTucker:
    def test_identical_is_one(self):
        a = np.array([0.3, -0.5, 0.8])
        assert axes.tucker(a, a) == pytest.approx(1.0)
        assert axes.tucker(a, 2.5 * a) == pytest.approx(1.0)  # scale-free

    def test_orthogonal_is_zero(self):
        assert axes.tucker([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_is_minus_one(self):
        a = np.array([1.0, 2.0])
        assert axes.tucker(a, -a) == pytest.approx(-1.0)


# ===================================================================
# cooking
# ===================================================================

class TestCooking:
    def test_ev2p_bounds(self):
        assert cooking.EV2P(1) == 0.125
        assert cooking.EV2P(7) == 0.875
        np.testing.assert_array_equal(cooking.EV2P(np.array([1.0, 4.0, 7.0])),
                                      [0.125, 0.5, 0.875])

    def test_base_rate_fit_recovers_known_rates(self):
        # construct B so that EV2P(B)[a, b] = P_true[b]: then the pairs
        # log-asymmetry is exactly l_b - l_a and the direct term pins level
        n = 6
        P_true = np.linspace(0.2, 0.8, n)
        B = np.tile(8.0 * P_true, (n, 1))  # constant columns, EVs in [1.6, 6.4]
        d_log = np.log(P_true)
        fit = cooking.base_rate_fit(B, d_log, lam=1.0)
        r = np.corrcoef(fit["P"], P_true)[0, 1]
        assert r > 0.95
        np.testing.assert_allclose(fit["P"], P_true, rtol=1e-6)
        assert set(fit) == {"l", "P", "psi", "phi"}

    def test_implied_phi_zero_diag_symmetric(self):
        rng = np.random.default_rng(4)
        n = 5
        Pc = rng.uniform(0.2, 0.8, size=(n, n))
        P = rng.uniform(0.2, 0.8, size=n)
        phi = cooking.implied_phi(Pc, P)
        np.testing.assert_array_equal(np.diag(phi), np.zeros(n))
        np.testing.assert_allclose(phi, phi.T, atol=1e-12)
        assert np.all(np.isfinite(phi))

    def test_column_center_offdiag_means_zero(self):
        rng = np.random.default_rng(6)
        B = rng.normal(size=(7, 7))
        C = cooking.column_center(B, skip_diag=True)
        mask = ~np.eye(7, dtype=bool)
        for j in range(7):
            assert abs(C[mask[:, j], j].mean()) < 1e-12
        # skip_diag=False centers the plain column means instead
        C2 = cooking.column_center(B, skip_diag=False)
        np.testing.assert_allclose(C2.mean(0), np.zeros(7), atol=1e-12)


# ===================================================================
# captions
# ===================================================================

class TestCaptions:
    MD = textwrap.dedent("""\
        # SELF
        **The self-report
        headline**

        First paragraph with {n_self} models
        wrapping over lines.

        Second paragraph.

        # JUDGE
        **Judge headline**

        Only paragraph, keeps {X} braces.
        """)

    def test_parse_two_sections(self, tmp_path):
        p = tmp_path / "caps.md"
        p.write_text(self.MD)
        caps = captions.load(str(p))
        assert set(caps) == {"SELF", "JUDGE"}
        assert caps["SELF"]["title"] == "The self-report headline"
        assert caps["SELF"]["paras"] == [
            "First paragraph with {n_self} models wrapping over lines.",
            "Second paragraph."]
        assert caps["SELF"]["sub"] == " ".join(caps["SELF"]["paras"])
        assert caps["JUDGE"]["title"] == "Judge headline"
        assert caps["JUDGE"]["paras"] == ["Only paragraph, keeps {X} braces."]

    def test_missing_title_raises(self, tmp_path):
        p = tmp_path / "bad.md"
        p.write_text("# KEY\nno bold title here\n")
        with pytest.raises(ValueError, match="no \\*\\*title\\*\\*"):
            captions.load(str(p))

    def test_render_substitutes_known_passes_unknown(self):
        out = captions.render("saw {n_self} models and {X} and {n_judge}",
                              {"n_self": 64, "n_judge": "ten"})
        assert out == "saw 64 models and {X} and ten"


# ===================================================================
# facets + load — real repo data, read-only
# ===================================================================

def _need(p, what):
    if not Path(p).exists():
        pytest.skip(f"{what} not present: {p}")


class TestLoadData:
    def test_adjectives_canonical(self):
        _need(paths.HUMAN_CORR, "human corr matrix")
        adj = __import__("pkit").load.adjectives()
        assert len(adj) == 525
        assert len(set(adj)) == 525
        assert all(a == a.lower() for a in adj)

    def test_human_corr_shape(self):
        _need(paths.HUMAN_CORR, "human corr matrix")
        from pkit import load
        H = load.human_corr()
        assert H.shape == (525, 525)
        np.testing.assert_allclose(np.diag(H.values), np.ones(525), atol=1e-9)

    def test_load_self_phi4(self):
        _need(paths.HUMAN_CORR, "human corr matrix")
        from pkit import load
        try:
            df = load.load_self("phi4")
        except FileNotFoundError:
            pytest.skip("phi4 selfreport file not present")
        counts = df.groupby("framing").size()
        assert sorted(counts.index) == sorted(load.FRAMINGS)
        assert (counts == 525).all()
        assert len(df) == 6 * 525
        assert np.isfinite(df["ev"]).all()
        assert df["ev"].between(1, 7).all()


@pytest.fixture(scope="module")
def clusters():
    _need(paths.HUMAN_CORR, "human corr matrix")
    _need(paths.ROOT / "instruments" / "trait_blocks_44.json",
          "trait_blocks_44.json")
    from pkit import facets
    return facets.clusters("blocks44")


class TestFacets:
    def test_clusters_valid_idx(self, clusters):
        assert len(clusters) == 44
        seen = set()
        for c in clusters:
            assert len(c["idx"]) == len(c["members"]) >= 2
            assert all(0 <= i < 525 for i in c["idx"])
            assert not (set(c["idx"]) & seen)  # blocks are disjoint
            seen.update(c["idx"])
            assert c["label"] in c["members"]

    def test_block_aggregation_structure(self, clusters):
        # block-level "identity": 1 within a cluster's block, 0 across ->
        # block() must return ~identity at the cluster level (its diagonal
        # uses off-diagonal-only means, so plain np.eye would give 0s)
        from pkit import facets
        S = np.zeros((525, 525))
        for c in clusters:
            S[np.ix_(c["idx"], c["idx"])] = 1.0
        Bl = facets.block(S, clusters)
        assert Bl.shape == (44, 44)
        np.testing.assert_allclose(np.diag(Bl.values), np.ones(44), atol=1e-12)
        off = Bl.values[~np.eye(44, dtype=bool)]
        np.testing.assert_allclose(off, np.zeros(off.size), atol=1e-12)


@pytest.mark.skipif(not paths.HUMAN_POR.exists(),
                    reason="gitignored data/ .por deposit not present")
class TestLoadHuman:
    def test_load_human_shape(self):
        pytest.importorskip("pyreadstat")
        from pkit import load
        M, labels = load.load_human()
        assert M.shape[1] == len(labels) == 525
        assert M.shape[0] > 100
        assert np.all(np.isfinite(M))
