from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from transformers import AutoTokenizer
from trl.trainer import ConstantLengthDataset
from transformers import TrainingArguments
from transformers import Trainer
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from trl import DataCollatorForCompletionOnlyLM
import numpy as np
from datasets import load_dataset
import wandb


debug_mode=0


if debug_mode==1:
    # Initialize wandb
    wandb.login()

# Load the train and validation datasets
train_dataset= load_dataset("Anthropic/hh-rlhf",split="train")
test_dataset = load_dataset("Anthropic/hh-rlhf",split="test")
# Take a subset of 800 examples from the training set for simplicity
train_dataset = train_dataset .select(range(800))
# Take a subset of 200 examples from the validation set for simplicity
test_dataset = test_dataset.select(range(200))




from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base") 



def custom_collate_fn(batch):
    chosen_texts = [item['chosen'] for item in batch]
    rejected_texts = [item['rejected'] for item in batch]
    
    chosen_encodings = tokenizer(chosen_texts, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
    rejected_encodings = tokenizer(rejected_texts, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
    
    return {
        "input_ids_chosen": chosen_encodings['input_ids'],
        "attention_mask_chosen": chosen_encodings['attention_mask'],
        "input_ids_rejected": rejected_encodings['input_ids'],
        "attention_mask_rejected": rejected_encodings['attention_mask'],
    }
     

iterator = iter( train_dataset)
one_sample = next( iterator )
print(list(one_sample.keys()))


######## Initialize the Model

from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained( "microsoft/deberta-v3-base", num_labels=1 )

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="RM_yck",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.001,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",  
    gradient_accumulation_steps=1,
    bf16=False,
    fp16=True,
    logging_strategy="steps",
    logging_steps=10,
    optim="adamw_hf",
    lr_scheduler_type="linear",
    ddp_find_unused_parameters=False,
    run_name="reward-hh_rlhf",
    report_to="wandb",
     load_best_model_at_end=False,
)


from trl import RewardTrainer

class CustomRewardTrainer(RewardTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_norms = {}
    
    
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=custom_collate_fn,
            shuffle=True,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):  # active when debug_mode=1
        return torch.utils.data.DataLoader(
            eval_dataset or self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=custom_collate_fn,
        )
    
    
    def training_step(self, model, inputs): # active when debug_mode=1
        # Perform the regular training step
        loss = super().training_step(model, inputs)

        # Log gradient norms for debugging
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradient_norms[f"gradient_norm/{name}"] = param.grad.norm().item()

        # Log the gradient norms to wandb
        wandb.log(self.gradient_norms)

        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None):
       
        # Perform the regular evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys)

        # Log a sample of predictions for debugging
        if eval_dataset is not None:
            sample_size = min(5, len(eval_dataset))
            samples = eval_dataset.select(range(sample_size))
            predictions = self.predict(samples)
            
            for i, (sample, prediction) in enumerate(zip(samples, predictions.predictions)):
                wandb.log({
                    f"sample_{i}/chosen": sample['chosen'],
                    f"sample_{i}/rejected": sample['rejected'],
                    f"sample_{i}/prediction": prediction[0]
                })

        return metrics
    
trainer = CustomRewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    max_length=256
)


if debug_mode==1:
    # Start a new wandb run
    with wandb.init(project="reward-model-training", name="reward-hh_rlhf"):
        # Train the model
        trainer.train()
        # Log the final model to wandb
        wandb.save("./RewardModel-final/*")
    # Close the wandb run
    wandb.finish()
else:
    trainer.train()


# Save the model
trainer.save_model("./RewardModel-final")


