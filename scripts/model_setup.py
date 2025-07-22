from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_name,nr_labels,label_2_id,id_2_label):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = nr_labels,
        label2id = label_2_id,
        id2label = id_2_label
    )

    return tokenizer,model