"""Task 3: training.

Spec:
  - Teacher forcing: decoder input = tgt[:, :-1], prediction target = tgt[:, 1:].
  - Loss: cross-entropy, label smoothing 0.1, ignore <pad>.
  - Optimiser: Adam, beta1 = 0.9, beta2 = 0.98, eps = 1e-9.
  - LR schedule (paper Eq. 3), warmup_steps = 4000:
        lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
  - Batch size 64, 20 epochs. Each epoch log train loss, dev loss and the
    current LR; save the checkpoint with the lowest dev loss.
  - Dev is only for checkpoint selection; test is used once, at the very end.
"""

# TODO: implement
