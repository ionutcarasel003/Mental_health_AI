import pandas as pd

def load_datasets(train_path,test_path,valid_path):
    #incarcarea datelor

    train_dfile = pd.read_csv(train_path, sep=';', names=["text", "label"], header=None)
    print("Train columns:", train_dfile.columns)

    test_dfile = pd.read_csv(test_path, sep=';', names=["text", "label"], header=None)
    print("Test columns:", test_dfile.columns)

    valid_dfile = pd.read_csv(valid_path, sep=';', names=["text", "label"], header=None)
    print("Validation columns:", valid_dfile.columns)

    #encodingul label-urilor

    labels = sorted(train_dfile['label'].unique())
    label_2_id = {label:i for i, label in enumerate(labels)}
    id_2_label = {i: label for label, i in label_2_id.items()}

    #maparea label-urilor

    for data_file in [train_dfile,valid_dfile,test_dfile]:
        data_file['label'] = data_file['label'].map(label_2_id)

    return train_dfile,valid_dfile,test_dfile,label_2_id,id_2_label