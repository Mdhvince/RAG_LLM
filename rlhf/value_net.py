import torch
from torch import nn as nn


class ValueNet(nn.Module):
    def __init__(self) -> None:
        """
        The value network is a simple neural network that outputs the value of the state.
        It will be represented as follows:

        ValueHead(
          (dropout): Dropout(p=0.1, inplace=False)
          (summary): Linear(in_features=768, out_features=1, bias=True)
          (flatten): Flatten(start_dim=1, end_dim=-1)
        )
        """
        super(ValueNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 1, bias=True),
            nn.Flatten()
        )

    def _format(self, input_ids):
        """
        Convert state to tensor if not and shape it correctly for the training process
        """
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=self.device, dtype=torch.float32)
            input_ids = input_ids.unsqueeze(0)
        return input_ids


    def forward(self, input_ids):
        """
        Forward pass through the model

        :param input_ids: The state of the environment -> the current ids of the tokens completion (prompt + generated)
        """
        input_ids = self._format(input_ids)
        value = self.model(input_ids)
        return value
