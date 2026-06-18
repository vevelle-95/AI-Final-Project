import torch
from sklearn.metrics import classification_report

def evaluate(model, dataloader, device):

    model.eval()
    preds, labels = [], []

    device_type = "cuda" if "cuda" in str(device) else "cpu"

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)
            image_mask = batch["image_mask"].to(device)


            labels_batch = batch["label"].cpu().numpy()

            with torch.amp.autocast(device_type=device_type, enabled=False):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    image_mask=image_mask
                )

            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if isinstance(logits, tuple):
                logits = logits[0]

            pred = outputs.argmax(dim=1).cpu().numpy()

            preds.extend(pred)
            labels.extend(labels_batch)

    print("\n================ EVALUATION REPORT ================")
    # Add labels=[0, 1, 2] to anchor the 3 target names properly
    print(classification_report(
        labels, 
        preds, 
        labels=[0, 1, 2],                             # <-- ADD THIS LINE
        target_names=["negative", "neutral", "positive"], 
        zero_division=0
    ))
    print("===================================================")