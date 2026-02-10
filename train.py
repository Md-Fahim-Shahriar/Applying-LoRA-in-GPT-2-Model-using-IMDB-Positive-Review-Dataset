from transformers import Trainer, TrainingArguments
from model import setup_lora_model
from data_utils import data_process
import os

print("Preparing the mode and dataset...")
model = setup_lora_model()
full_dataset = data_process(review=500)


args = TrainingArguments(
    output_dir="./results",
    num_train_epochs = 3,
    logging_dir= "./logs",
    logging_steps= 5,
    
    eval_strategy= "steps",
    eval_steps = 20,

    save_strategy="epoch",
    learning_rate= 2e-4,
    report_to = "tensorboard"
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset= full_dataset["train"],
    eval_dataset= full_dataset["test"]
)

print("Starting Training..")
trainer.train()

model.save_pretrained("./final_lora_model")

print("Training Complete, Model saved to './final_lora_model'")