# Asymmetry Report

baseline(no_injection) asym_L median range: min=0.5439, median=0.8796, max=1.0255
mechanism asym_L median range: min=0.6372, median=0.8209, max=1.0158
symmetric_control_v2 asym_L median range: min=0.4741, median=0.8690, max=1.0698

Interpretation:
If baseline asym_L medians are materially above 0, L already carries directionality prior to injection.

Measured object definition:
- L: raw attention logits (QK^T/sqrt(d_head)) at probe layer 2, pre-bias, pre-mask, pre-clamp; heads averaged.
- U: raw u_q/u_k pathway at probe layer 2, pre-clamp.
- Mask: attn_mask is applied inside attention, but asymmetry uses logits_raw before masking; padded positions remain in L/U.
- Eval: eval_size=10000 per checkpoint; generator run_id seeded from ckpt seed(s): 0, 1, 2.
