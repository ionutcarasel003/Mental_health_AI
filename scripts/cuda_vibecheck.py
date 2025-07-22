import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def check_cuda():
    print("Checking CUDA availability...\n")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU device: {device_name}")
    else:
        print("⚠️ CUDA not available. Will use CPU.")

def load_model():
    model_name = "distilbert-base-uncased"
    print(f"\nLoading tokenizer and model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model loaded on device: {device}")
    return tokenizer, model

def run_dummy_inference(tokenizer, model):
    print("\nRunning dummy inference...")

    input_text = "This is a test for CUDA"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    print("Inference successful. Output logits:")
    print(outputs.logits)

if __name__ == "__main__":
    check_cuda()
    tokenizer, model = load_model()
    run_dummy_inference(tokenizer, model)
