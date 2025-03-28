import os
import json
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import pandas as pd
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Using bfloat16 for better training efficiency
    trust_remote_code=True
)

# Move model to device
model.to(device)

# 2. Load and prepare AfricMed MCQ data
# Try loading from CSV file
africmed_df = pd.read_csv("africmed_mcq.csv")
print(f"Loaded AfricMed MCQ data from CSV: {len(africmed_df)} rows")

# Parse the JSON in answer_options if it's stored as a string
if isinstance(africmed_df['answer_options'].iloc[0], str):
    africmed_df['answer_options'] = africmed_df['answer_options'].apply(json.loads)

# 3. Prepare data for training by creating suitable prompts and completions
def prepare_mcq_for_training(row):
    # Extract options from the answer_options dict or parse it if it's a string
    options = row['answer_options']
    
    # Format the options as a readable string
    formatted_options = "\n".join([f"{key}: {value}" for key, value in options.items()])
    
    # Create an instruction prompt
    instruction = f"Answer the following multiple-choice medical question correctly. Choose the most appropriate option."
    
    # Create the input prompt with the question and options
    input_text = f"{row['question_clean']}\n\n{formatted_options}"
    
    # The correct answer
    correct_answer_key = row['correct_answer']
    # correct_answer_text = options[correct_answer_key]
    
    # Format output to include both the option identifier and the text
    output = f"The correct answer is {correct_answer_key}"
    
    # Format in the style expected by Phi-3
    formatted_text = f"<|system|>\nYou are an expert medical AI assistant.\n<|user|>\n{instruction}\n\n{input_text}\n<|assistant|>\n{output}"
    
    return {"text": formatted_text}

# Apply the formatting function
formatted_data = africmed_df.apply(prepare_mcq_for_training, axis=1).tolist()
print(f"Formatted {len(formatted_data)} MCQ items for training")

# Split the data into training and validation sets
train_data, val_data = train_test_split(formatted_data, test_size=0.1, random_state=42)

# Create datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

africmed_data = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# 4. Tokenize the data
def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=4096,  # Phi-3-mini context window is 4k tokens
        padding="max_length",
        return_tensors="pt"
    )
    
    # For causal language modeling, we need the input_ids as labels too
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

# Apply tokenization
print("Tokenizing datasets...")
tokenized_datasets = africmed_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Remove original text column after tokenization
)

# 5. Set up data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We want causal language modeling, not masked
)

# 6. Define training arguments
training_args = TrainingArguments(
    output_dir="./pretrained-phi3-mini-africmed-mcq",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    eval_steps=100,
    save_steps=500,
    warmup_steps=100,
    evaluation_strategy="steps",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    gradient_accumulation_steps=4,   # To handle larger effective batch sizes
    learning_rate=5e-5,
    weight_decay=0.01,
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# 8. Train the model
print("Starting training...")
trainer.train()

# 9. Save the fine-tuned model
model_save_path = "./pretrained-phi3-mini-africmed-mcq-final"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# 10. Test the model with a sample medical MCQ
sample_question = "Which of the following infectious diseases is caused by the bacterium Mycobacterium leprae and primarily affects the skin, nerves, and mucous membranes?"
sample_options = {
    "option1": "Tuberculosis", 
    "option2": "Leprosy", 
    "option3": "Yaws", 
    "option4": "Buruli ulcer disease", 
    "option5": "Lymphatic filariasis"
}
formatted_options = "\n".join([f"{key}: {value}" for key, value in sample_options.items()])

# Format similar to training examples
test_prompt = f"<|system|>\nYou are an expert medical AI assistant.\n<|user|>\nAnswer the following multiple-choice medical question correctly. Choose the most appropriate option.\n\n{sample_question}\n\n{formatted_options}\n<|assistant|>\n"

inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

# Generate a response
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=4096,
        temperature=0.1,  # Lower temperature for more deterministic output
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nSample test:")
print(f"Question: {sample_question}")
print(f"Model response: {response.split('<|assistant|>')[-1].strip()}")