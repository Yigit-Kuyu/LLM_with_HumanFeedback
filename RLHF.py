from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import load_dataset
import torch




# Load the squad_v2 dataset same as in SFT
train_dataset = load_dataset("rajpurkar/squad_v2", split="train")

# Take a subset if needed (for faster iteration during development)
train_dataset = train_dataset.select(range(1000))  # Adjust the number as needed


# Load the SFT model
sft_model = AutoModelForCausalLM.from_pretrained("./SFTModel-final")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load the Reward Model
reward_model = AutoModelForSequenceClassification.from_pretrained("./RewardModel-final")



# PPO configuration
ppo_config = PPOConfig(
    model_name="./PPOOModel-final",
    learning_rate=1e-5,
    batch_size=1,  # Set batch size to 1 for simplicity
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1, #KL divergence
    ppo_epochs=4,
    seed=42,
)

# Prepare the model for PPO training adding value-head to predict the expected reward (value) for a given input.
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)


def prepare_ppo_dataset(example):
    return {
        "query": f"Answer the following question based on the given context.\n\nQuestion: {example['question']}\n\nContext: {example['context']}\n\nAnswer:"
    }

train_dataset = train_dataset.map(prepare_ppo_dataset, remove_columns=train_dataset.column_names)


device = "cuda" if torch.cuda.is_available() else "cpu"
ppo_model = ppo_model.to(device)
reward_model = reward_model.to(device)


# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=ppo_model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=train_dataset,  
    data_collator=None,
)

####### Training PPO

num_epochs = 3 # just for simplicity
for epoch in range(num_epochs):
    for batch in ppo_trainer.dataloader:
        query = batch["query"][0]  # Assuming batch size 1 for simplicity
        
        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = inputs["input_ids"].squeeze(0)  # Remove batch dimension
        
        # Check shape of input_ids
        #print(f"input_ids shape: {input_ids.shape}")
        
        
        # Generate response using ppo_trainer.generate
        response_ids = ppo_trainer.generate(
            [input_ids],
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
         
        #print(f"Raw response: {response}")
        
        # Remove the prompt from the response
        response = response[len(query):].strip()
        
        #print(f"Trimmed response: {response}")
        
        # Check if response is empty
        if not response:
            print("Empty response generated. Skipping this example.")
            continue 
        
        
        
        # Get reward from the reward model
        inputs = tokenizer(query + "  " + response, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            reward = reward_model(**inputs).logits.item()
        
        # Convert queries and responses to tensors
        query_tensor = input_ids.long()
        response_tensor = tokenizer(response, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0).to(device).long()
        reward_tensor = torch.tensor([reward], dtype=torch.float, device=device)

       
        
        # Run PPO step with lists of tensors
        try:
            stats = ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor]) #  Computing the advantage, updating the policy, updating the value function 
            print(f"Epoch {epoch}, Reward: {reward}")
            print(f"Query: {query}")
            print(f"Response: {response}")
            #print(f"Stats: {stats}")
        except Exception as e:
            print(f"Error during PPO step: {e}")
            print(f"Query tensor: {query_tensor}")
            print(f"Response tensor: {response_tensor}")
            print(f"Reward tensor: {reward_tensor}")
            continue
        

        


        
# Save the final model
ppo_trainer.save_pretrained("./PPO-finetuned-model")


