from datasets import load_dataset
from transformers import AutoTokenizer


def dataset_checking():
    raw_dataset = load_dataset("imdb")

    positive_reviews = raw_dataset["train"].filter(lambda x: x["label"] == 1)

    

    pos_indices = [i for i, x in enumerate(raw_dataset["train"]) if x["label"] == 1]


    sample = raw_dataset["train"][1250]
    #print(sample)

    if sample["label"] == 1:
        sentiment = "Positive"


    else:
        sentiment = "Negative"
    
    sample_final = sample["text"][:50]
    
    return sentiment, sample_final, positive_reviews, pos_indices

def tokenize_function(examples):
    
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    outputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    outputs["labels"] = [list(ids) for ids in outputs["input_ids"]]
    
    return outputs

def data_process(review = 500):

    _, _, positive_reviews, _ = dataset_checking() 

    small_pos_dataset = positive_reviews.select(range(review))
    split_dataset = small_pos_dataset.train_test_split(test_size = 0.2)

    tokenized_dataset = split_dataset.map(
    tokenize_function,
    batched= True,
    remove_columns=split_dataset["train"].column_names
    )
    
    return tokenized_dataset

    

    

if __name__ == "__main__":

    tokenized_dataset_checking = data_process(10)

    print("-"*20)
    print(tokenized_dataset_checking)

    # n = int(input("Enter the number of reviews: "))
    # tokenized_dataset = data_process(n)
    # print(tokenized_dataset)

    for i in range(3):
        print(f"Review #{i}")
        # print(f"Text: {tokenized_dataset_checking['train'][i]['text'][:50]}")
        print(f"Token ID: {tokenized_dataset_checking['train'][i]['input_ids'][:50]}")
        print(f"Token Labels: {tokenized_dataset_checking['train'][i]['labels'][:50]}")

    # print(f"The postive reviews are: {test}")

    # sentiment, data, positive_reviews, pos_indices = dataset_checking(12500)

    # print(f"---Sample review (Sentiment: {sentiment})---")
    # print(data)
    # print(f"\n Total Number of positive reviews are: {positive_reviews}")
    # print(f"\n The positive reviews index starts from: {pos_indices[0]}")




    