import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, peft_model, generation_config) -> None:
        """
        The policy network is an LLMPeftModel that outputs the ids of the tokens generated.

        :param peft_model: The peft_model model (finetuned)
        :param generation_config: The generation config used for the PEFT model
        """
        super(PolicyNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_config = generation_config
        self.model = peft_model

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
        outputs = self.model.generate(input_ids=input_ids, generation_config=self.gen_config)
        logits = outputs.logits

        token_output = outputs[0]  # tokens generated, will be decoded and added to the next prompt-completion

        # The forward will need to be called once. Then we will iterate over the tokens and logits generated to
        # simulate the fact that the model is generating one token at a time, hence get one reward at a time.

        return logits, token_output
