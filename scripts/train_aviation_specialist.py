import torch
import os
import sys
import json
import numpy as np
import evaluate
from PIL import Image
from datasets import Dataset, Features, ClassLabel, Image as DatasetImage
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator,
    EarlyStoppingCallback
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ColorJitter
)

# --- CONFIGURATION ---
DATA_PATH = r"D:\Development\AviationDataSet" 
# We use the high-resolution 384px ViT as our foundation
MODEL_ID = "google/vit-base-patch16-384"
OUTPUT_DIR = "./aviation_model_v3"

# --- GLOBAL TRANSFORMS (Required for Windows Multiprocessing) ---
# Hardcode 384px for high-resolution refinement
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
RESOLUTION = 384
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

_train_transforms = Compose([
    RandomResizedCrop(RESOLUTION),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ToTensor(),
    normalize,
])

_val_transforms = Compose([
    Resize(RESOLUTION),
    CenterCrop(RESOLUTION),
    ToTensor(),
    normalize,
])

def preprocess_train(examples):
    # This function must be at the global scope for Windows workers to pickle it
    examples["pixel_values"] = [_train_transforms(img.convert("RGB")) for img in examples["image"]]
    if "image" in examples:
        del examples["image"]
    return examples

def preprocess_val(examples):
    # This function must be at the global scope for Windows workers to pickle it
    examples["pixel_values"] = [_val_transforms(img.convert("RGB")) for img in examples["image"]]
    if "image" in examples:
        del examples["image"]
    return examples

def train():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: DATA_PATH not found at {DATA_PATH}")
        return

    print(f"--- Scanning Dataset: {DATA_PATH} ---")
    
    class_names = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    if not class_names:
        print("ERROR: No subdirectories found in DATA_PATH!")
        return

    image_paths = []
    labels = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_PATH, class_name)
        count = 0
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, f))
                labels.append(label_idx)
                count += 1
        if count > 0:
            print(f"  [+] Found {count} images for: {class_name}")

    if not image_paths:
        print("ERROR: No images found!")
        return

    print(f"\nTotal images: {len(image_paths)} | Classes: {len(class_names)}")

    # 2. Create Dataset
    features = Features({"image": DatasetImage(), "label": ClassLabel(names=class_names)})
    raw_dataset = Dataset.from_dict({"image": image_paths, "label": labels}, features=features)
    dataset = raw_dataset.train_test_split(test_size=0.15, seed=42) # Increased val size
    
    id2label = {str(i): label for i, label in enumerate(class_names)}
    label2id = {label: str(i) for i, label in enumerate(class_names)}

    # Use with_transform for on-the-fly processing (Fixes RAM issues on Windows)
    train_ds = dataset["train"].with_transform(preprocess_train)
    val_ds = dataset["test"].with_transform(preprocess_val)

    # 4. Load Model
    print(f"\n--- Initializing Model from Foundation: {MODEL_ID} ---")
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True 
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False, 
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=15,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        tf32=True if torch.cuda.is_available() else False,
    )

    # 6. Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    # 7. Run Training
    print("\n--- Starting Fine-Tuning (Combining Dataset + Base Knowledge) ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    
    print("\n--- Evaluating Best Model ---")
    eval_results = trainer.evaluate()
    print(f"Final Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    
    trainer.save_model()
    trainer.save_state() # Save trainer_state.json
    processor.save_pretrained(OUTPUT_DIR)
    
    # Save metrics to a file
    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
        
    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(class_names))
        
    print(f"\nSUCCESS! Enhanced Model saved to {OUTPUT_DIR}")
    print(f"You can now run 'python scripts/plot_training_history.py' to see the performance curves.")

if __name__ == "__main__":
    train()
