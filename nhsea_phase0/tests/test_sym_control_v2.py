import numpy as np
import torch
from torch.utils.data import DataLoader

from nhsea.data import DatasetConfig, ForwardChainDataset, collate_batch
from nhsea.generators import ForwardChainConfig
from nhsea.model import ModelConfig, TinyTransformer
from nhsea.operator import OperatorSpec, build_run_operator, scale_norms


def _gamma_from_norms(A_vals, B0_vals):
    A_median = float(np.median(np.asarray(A_vals))) if A_vals else 0.0
    B0_median = float(np.median(np.asarray(B0_vals))) if B0_vals else 0.0
    gamma = 1.0 if B0_median == 0.0 else A_median / B0_median
    return gamma, A_median, B0_median


def test_gamma_median_match_on_small_batch():
    U1 = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    U2 = np.array([[0.0, -3.0], [4.0, 0.0]], dtype=np.float64)
    A_vals, B0_vals = [], []
    for U in (U1, U2):
        A, B0 = scale_norms(U, beta=1.0)
        A_vals.append(A)
        B0_vals.append(B0)
    gamma, A_median, _B0_median = _gamma_from_norms(A_vals, B0_vals)
    scaled_B0 = [gamma * b0 for b0 in B0_vals]
    assert np.isclose(np.median(scaled_B0), A_median)


def test_v2_control_does_not_change_mechanism_output():
    L = np.array([[0.0, 0.2], [0.3, 0.0]], dtype=np.float64)
    U = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    spec_a = OperatorSpec(alpha=0.1, beta=0.2, variant="mechanism")
    spec_b = OperatorSpec(alpha=0.1, beta=0.2, gamma=3.0, variant="mechanism")
    O_a = build_run_operator(L, U, spec_a)
    O_b = build_run_operator(L, U, spec_b)
    assert np.allclose(O_a, O_b)


def test_gamma_deterministic_with_fixed_seed_and_eval_set():
    def compute_gamma(seed: int) -> float:
        torch.manual_seed(seed)
        np.random.seed(seed)
        gen_cfg = ForwardChainConfig()
        data_cfg = DatasetConfig(
            task="forward",
            split="eval",
            size=4,
            seed=seed,
            T=gen_cfg.T,
            vocab_size=gen_cfg.vocab_size,
        )
        dataset = ForwardChainDataset(data_cfg, gen_cfg)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_batch)

        model_cfg = ModelConfig(vocab_size=len(dataset.vocab), T=gen_cfg.T)
        model = TinyTransformer(model_cfg, probe_layer=2)
        model.set_num_classes(2)
        model.eval()

        A_vals, B0_vals = [], []
        with torch.no_grad():
            for input_ids, attn_mask, _labels, _metas in loader:
                _logits, _probe_logits, probe_U = model(
                    input_ids,
                    attn_mask=attn_mask,
                    variant="symmetric_control_v2_normmatched",
                    alpha=0.5,
                    beta=1.0,
                    gamma=1.0,
                    return_probe=True,
                )
                U_np = probe_U.numpy()
                for U in U_np:
                    A, B0 = scale_norms(U, beta=1.0)
                    A_vals.append(A)
                    B0_vals.append(B0)
        gamma, _A_median, _B0_median = _gamma_from_norms(A_vals, B0_vals)
        return float(gamma)

    gamma1 = compute_gamma(0)
    gamma2 = compute_gamma(0)
    assert np.isclose(gamma1, gamma2)
