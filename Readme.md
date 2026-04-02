# Utility of Synthetic Images from Diffusion Models for Data Augmentation

We will explore the positives and negatives of synthetic data.

- the postiives will be explored by replicating technique of resotring class imbalance through DA-Fusion by Trabucco et al. [https://arxiv.org/pdf/2302.07944]

  - There exists powerful generative models such as Gemini's Nano banana 3 and Dalle.3 from OpenAI, which could replace the diffusion model for synthetic data generation. These outputs would yield hgihger fidelity, but would be comparitively very costly. we aim to show that a simple Stable diffusion v1.5 baseline model can indeed improve performance on a downstream RestNet model. This result would suggest that if we spent some time finetuning our diffusion model, it could be useful for a more niche downstream classification task.
  - another potential research "branch" here is, if we further supply non-photorealistic synthetic samples as training instances for our downstream task, how does this impact ResNet's performance? These samples could be artistic depictions of the class "Cat" instead of photorealistic images of cats. Do these samples help generalisation becuse the non-photorealistic samples, though very noisy, still encode some degree of meaning of the representative class. if this does result in good performance then, we could have even more training samples produced, simply by using another open source finetuned model. We might also conclude that the features that ou
  - but if performance drops dramatically, then it is an interesting conclusion.
- the negatives will be explored by highlighting specific cases of data introduced bias. will look into the papers:

  - Biased Generalization in Diffusion Models
  - **A**NTI-EXPOSURE **B**IAS IN **D**IFFUSION **M**ODELS
