## INFO

The framework consists of three steps, first: supervised fine-tuning (SFT), second: reward model creation (RM), and last: PPO implementation with human feedback data (RLHF).

- **SFT.py:** This code fine-tunes a large language model (GPT-Neo 1.3B) on a question-answering task using the SQuAD v2 dataset. It prepares the data by combining questions, contexts, and answers into a single format. The model is set up with memory-efficient techniques like quantization and LoRA. The code then trains the model using the Hugging Face Trainer, which handles the training loop and evaluation. The goal is to improve the model's ability to answer questions based on given contexts, creating a specialized version of the original language model for this specific task.
-  **RewardModel.py:** This code implements a reward model for RLHF. It uses the Anthropic/hh-rlhf dataset, which contains pairs of chosen and rejected responses. The model, based on DeBERTa-v3, is trained to predict a scalar reward value to distinguish between preferred and non-preferred responses. The training process includes options for debugging and logging with Weights&Biases (wandb).
-  **RLHF.py:** This code implements the RLHF stage using PPO. It loads a pre-trained SFT model and a reward model, then prepares them for PPO training. The SQuAD v2 dataset is used, preprocessed for the RLHF task. A PPO configuration is set up with parameters including learning rate, batch size, and KL divergence target. The process aims to fine-tune the language model to maximize rewards while maintaining similarity to the original model through KL divergence constraint. 

The basic flow of the framework can be seen below:

![AV_Module](https://github.com/Yigit-Kuyu/LLM_with_HumanFeedback/blob/main/Frameworks.jpg)
