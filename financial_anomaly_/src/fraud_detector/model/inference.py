import torch

from fraud_detector.schemas import Transaction

MODEL_VERSION = "baseline-v1"


class FraudScorer:
    """Deterministic baseline scorer with a PyTorch inference boundary."""

    def __init__(self) -> None:
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 1), torch.nn.Sigmoid())
        with torch.no_grad():
            self.model[0].weight.copy_(torch.tensor([[1.2, -0.4, 0.08, 0.9]]))
            self.model[0].bias.fill_(-1.0)
        self.model.eval()

    @staticmethod
    def _scale(transaction: Transaction) -> torch.Tensor:
        values = [
            transaction.amount / 1000.0,
            transaction.account_age_days / 365.0,
            transaction.transactions_last_hour / 10.0,
            float(transaction.is_international),
        ]
        return torch.tensor([values], dtype=torch.float32)

    def predict(self, transaction: Transaction) -> float:
        with torch.inference_mode():
            return float(self.model(self._scale(transaction)).item())
