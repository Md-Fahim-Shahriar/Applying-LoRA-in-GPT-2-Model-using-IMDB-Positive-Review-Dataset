from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


base_model_path = "gpt2"
lora_model_path = "./final_lora_model"


tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token


print("Loading model... this may take a moment.")
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, lora_model_path)


device = 0 if torch.cuda.is_available() else -1


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)


prompt = "The movie was "
print(f"\nPrompt: {prompt}")

results = generator(
    prompt,
    max_length=50,
    num_return_sequences=3, 
    pad_token_id=tokenizer.eos_token_id,
    truncation=True
)

print("\n--- Model Generations ---")
for i, res in enumerate(results):
    print(f"Result {i+1}: {res['generated_text']}")