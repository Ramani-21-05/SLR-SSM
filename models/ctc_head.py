"""
CTC classification head.

Same loss formulation as STNet (Eq. 6 in the paper), so WER stays directly
comparable to their reported numbers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, blank_id: int = 0):
        super().__init__()
        self.classifier = nn.Linear(d_model, vocab_size)
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> log-probs (T, B, vocab_size), CTC's expected layout."""
        logits = self.classifier(x)                 # (B, T, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)             # (T, B, vocab)

    def compute_loss(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs:      (T, B, vocab) from forward()
        targets:         1D concatenated target label sequence
        input_lengths:   (B,) actual (unpadded) frame counts per sample
        target_lengths:  (B,) actual gloss-sequence lengths per sample
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    @torch.no_grad()
    def greedy_decode(self, log_probs: torch.Tensor):
        """
        Simple greedy CTC decode (argmax + collapse repeats + drop blanks).
        Swap for beam search decoding for reported WER numbers if needed.
        log_probs: (T, B, vocab) -> list of B decoded id sequences.
        """
        preds = log_probs.argmax(dim=-1).transpose(0, 1)  # (B, T)
        decoded = []
        for seq in preds:
            out, prev = [], None
            for tok in seq.tolist():
                if tok != self.blank_id and tok != prev:
                    out.append(tok)
                prev = tok
            decoded.append(out)
        return decoded
