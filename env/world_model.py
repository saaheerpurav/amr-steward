# SAAHEER OWNS THIS FILE
# JEPA-inspired world model for diagnostic information gain prediction.
# Pre-trained separately, frozen during GRPO training.
# TODO Saaheer: implement AMRWorldModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

AVAILABLE_TOOLS = [
    "interpret_resistance_meropenem",
    "interpret_resistance_ceftazidime_avibactam",
    "interpret_resistance_colistin",
    "interpret_resistance_vancomycin",
    "interpret_resistance_ceftriaxone",
    "interpret_resistance_piperacillin_tazobactam",
    "check_guideline_bacteremia",
    "check_guideline_UTI",
    "check_guideline_pneumonia",
    "check_guideline_intra_abdominal",
    "assess_patient_factors",
    "interpret_resistance_tigecycline",
]
NUM_TOOLS = len(AVAILABLE_TOOLS)
TOOL_TO_IDX = {t: i for i, t in enumerate(AVAILABLE_TOOLS)}


class AMRWorldModel(nn.Module):
    """JEPA-inspired world model.
    Predicts information gain of running each diagnostic test
    given what the agent already knows."""

    def __init__(self, state_dim: int = 64, repr_dim: int = 128, ema_decay: float = 0.99):
        super().__init__()
        self.ema_decay = ema_decay

        # Context encoder: known state → abstract representation
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, repr_dim),
        )

        # Target encoder: EMA copy, not trained by backprop
        self.target_encoder = deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: context repr + test one-hot → predicted target repr
        self.predictor = nn.Sequential(
            nn.Linear(repr_dim + NUM_TOOLS, 256),
            nn.ReLU(),
            nn.Linear(256, repr_dim),
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder."""
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_p.data = self.ema_decay * tgt_p.data + (1 - self.ema_decay) * ctx_p.data

    def forward(self, known_state: torch.Tensor, tool_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_repr, target_repr) for JEPA loss computation."""
        ctx_repr = self.context_encoder(known_state)
        tool_onehot = F.one_hot(tool_idx, num_classes=NUM_TOOLS).float()
        pred_repr = self.predictor(torch.cat([ctx_repr, tool_onehot], dim=-1))
        with torch.no_grad():
            tgt_repr = self.target_encoder(known_state)
        return pred_repr, tgt_repr

    def predict_information_gain(self, known_state: torch.Tensor, tool_name: str) -> float:
        """Returns estimated information gain (0-1) for running a specific tool."""
        tool_idx = torch.tensor(TOOL_TO_IDX.get(tool_name, 0))
        pred_repr, tgt_repr = self.forward(known_state.unsqueeze(0), tool_idx.unsqueeze(0))
        # Information gain = how different predicted state is from current
        gain = 1.0 - F.cosine_similarity(pred_repr, tgt_repr, dim=-1).item()
        return float(gain)

    def get_test_rankings(self, known_state: torch.Tensor, available_tools: list[str]) -> list[tuple[str, float]]:
        """Returns tools sorted by predicted information gain (highest first)."""
        scores = [(tool, self.predict_information_gain(known_state, tool)) for tool in available_tools]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def encode_known_state(self, tool_results: list[str], patient_features: dict) -> torch.Tensor:
        """Convert tool results + patient features into a fixed-dim state vector.
        TODO Saaheer: implement proper feature extraction."""
        # Stub: return zeros — replace with real feature encoding
        return torch.zeros(64)


def enrich_observation(base_obs_text: str, world_model: AMRWorldModel,
                        tool_results: list[str], patient_features: dict,
                        available_tools: list[str]) -> str:
    """Append JEPA information gain predictions to the observation text."""
    known_state = world_model.encode_known_state(tool_results, patient_features)
    rankings = world_model.get_test_rankings(known_state, available_tools)

    lines = ["", "PREDICTED INFORMATION GAIN (run highest first):"]
    for tool, score in rankings:
        lines.append(f"  - {tool}: {score:.2f}")

    return base_obs_text + "\n".join(lines)
