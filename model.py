from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def setup_lora_model(model_id = "gpt2"):

    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    config = LoraConfig(
        r = 8,
        lora_alpha = 32,
        target_modules = ["c_attn"],
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    model = get_peft_model(base_model, config)

    return model

if __name__ == "__main__":

    model = setup_lora_model()
    model.print_trainable_parameters()