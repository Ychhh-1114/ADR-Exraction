from transformers import BertTokenizer


if __name__ == '__main__':

    text = "Immobilization, while Paget's bone disease was present, and perhaps enhanced activation of dihydrotachysterol by rifampicin, could have led to increased calcium-release into the circulation."
    tokenizer = BertTokenizer.from_pretrained("./bert-model/biobert-base-cased-v1.2")
    tokened = tokenizer(text)
    print(tokened)
    print(tokenizer.decode(tokened["input_ids"]))