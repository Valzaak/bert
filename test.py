import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

wnut = load_dataset("wnut_17")

model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)


test = wnut["test"]


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Apply the function to the test dataset
tokenized_test = test.map(tokenize_and_align_labels, batched=True)
preds_list = []
true_labels_list = []

# Now you can use tokenized_test in your model
with torch.no_grad():
    for i in tqdm(range(len(tokenized_test))):
        input = {
            key: torch.tensor(tokenized_test[i][key]).unsqueeze(0)
            for key in ["input_ids", "token_type_ids", "attention_mask", "labels"]
        }
        logits = model(**input).logits
        preds = torch.argmax(logits, dim=-1).numpy().flatten()
        true_labels = input["labels"].numpy().flatten()

        # Ignore special tokens
        mask = true_labels != -100
        preds = preds[mask]
        true_labels = true_labels[mask]

        preds_list.extend(preds)
        true_labels_list.extend(true_labels)
# Calculate metrics
f1 = f1_score(true_labels_list, preds_list, average="micro")
print(f"F1 Score: {f1}")
