from datasets import Dataset

def tokenize_i(i,tokenizer):

    return tokenizer(
        i["text"],
        truncation = True,
        padding = "max_length"
    )

def tokenize_and_wrap(tokenizer,df):
    df = df[["text","label"]].copy()
    dataset = Dataset.from_pandas(df)

    def apply_tokenizer(batch):
        return tokenize_i(batch,tokenizer)

    dataset = dataset.map(apply_tokenizer,batched = True)

    return dataset