"""
Parameter Efficient Fine-tuning

> Data annotation (Input and Target): Can be on single or multiple tasks.
> Fine-tuning with cross entropy loss (comparison of the probability distribution of tokens)

  - SELECTIVE METHOD: Update a subset of the parameters of the model by freezing most of the existing model weights.

  - ADDITIVE METHOD:
    Update the existing model weights by freezing them all and add additional parameters to it. Another Additive method
    is Prompt tuning (different from Prompt Engineering): We add additional trainable tokens (called soft prompt) then
    do supervised learning on them (perform very well for LLMs with +10B parameters).

  - RE-PARAMETERIZATION METHOD (i.e. LoRA - Low Rank Adaptation):
    Reduce the number of parameters to train by creating new low rank transformations of the original model weights.

> Evaluation using benchmarks.
"""