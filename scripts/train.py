from transformers import TrainingArguments, Trainer, logging, DataCollatorWithPadding
from load_data import load_datasets
from preprocessing import tokenize_and_wrap
from model_setup import load_model_and_tokenizer
from metrics import compute_metrics
from plot_callback import PlotMetricsCallback

logging.set_verbosity_error()

def main():
    model_name = "distilbert-base-uncased"

    # 1. Încărcare date
    df_train, df_val, df_test, label2id, id2label = load_datasets(
        "../Dataset/train.txt", "../Dataset/val.txt", "../Dataset/test.txt"
    )

    # 2. Încărcare tokenizer + model
    tokenizer, model = load_model_and_tokenizer(
        model_name, nr_labels=len(label2id), label_2_id=label2id, id_2_label=id2label
    )

    # 3. Preprocesare + transformare în HuggingFace Dataset
    train_dataset = tokenize_and_wrap(tokenizer, df_train)
    val_dataset = tokenize_and_wrap(tokenizer, df_val)

    # 4. Collator pentru padding dinamic (eficient pentru GPU)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Setări de antrenare
    training_args = TrainingArguments(
        output_dir="../model",
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        logging_dir="../logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"  
    )

    # 6. Trainer HF
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[PlotMetricsCallback()]
    )

    # 7. Antrenare + salvare model final
    trainer.train()
    trainer.save_model("../model")
    tokenizer.save_pretrained("../model")

if __name__ == "__main__":
    main()
