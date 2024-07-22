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


debug_mode=0



# Load the train and validation datasets from huggingface
train_dataset = load_dataset("rajpurkar/squad_v2", split="train")
val_dataset = load_dataset("rajpurkar/squad_v2", split="validation")

# Take a subset of 800 examples from the training set
train_dataset = train_dataset .select(range(800))
# Take a subset of 200 examples from the validation set
val_dataset = val_dataset.select(range(200))



if debug_mode:

    # Print the first example to see its structure
    print(train_dataset[0])
    print('####################################')
    # Print the keys of the first example
    print(train_dataset[0].keys())
    print('####################################')
    # If 'answers' is a key, let's look at its structure
    if 'answers' in train_dataset[0]:
        print(train_dataset[0]['answers'])# Print the first example



# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B") # Open-source alternative to GPT-3
tokenizer.pad_token = tokenizer.eos_token


# Define the response template
response_template = "Answer:"
def prepare_sample_text(question, context, answer):
    return f"Question: {question}\n\nContext: {context}\n\n{response_template} {answer}"

def tokenize_function(examples):
    questions = examples['question']
    contexts = examples['context']
    answers = [ans['text'][0] if ans['text'] else 'No answer available' for ans in examples['answers']]
    
    prepared_texts = [prepare_sample_text(q, c, a) for q, c, a in zip(questions, contexts, answers)]
    
    return tokenizer(
        prepared_texts,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

if debug_mode:
    # Test the tokenize_function
    print("Testing tokenize_function with a single example:")
    print(tokenize_function(train_dataset[0]))

    print("\nTesting tokenize_function with multiple examples:")
    print(tokenize_function(train_dataset[:5]))


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)


# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# Training arguments
training_args = TrainingArguments(
    output_dir="./sft_yck",
    evaluation_strategy="steps",
    save_strategy="steps",
    num_train_epochs=1,
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    gradient_accumulation_steps=4,
    fp16=True,
    weight_decay=0.05,
    run_name="sft_yck",
    report_to="wandb",
    load_best_model_at_end=False,
    
)


# Quantization configuration
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-1.3B", 
    quantization_config=quantization_config, 
    device_map="auto"
)


# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
# Prepare model for training
model.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)
# Ensure all parameters that should be trainable are set to require gradients
for name, param in model.named_parameters():
    if 'lora' in name or 'Lora' in name:
        param.requires_grad = True

# Print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)

# Prepare trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        response_template=response_template
    ),
)

# Debug: Check if any parameters require gradients
print("Parameters requiring gradients:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# Train
print("Training...")
trainer.train()

# Evaluate
print("Evaluating...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model
trainer.save_model("./SFTModel-final")