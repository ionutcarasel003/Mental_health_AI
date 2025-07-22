import transformers
print("Transformers version:", transformers.__version__)
print("Transformers location:", transformers.__file__)
print(transformers.TrainingArguments)

from transformers import TrainingArguments

help(TrainingArguments)
