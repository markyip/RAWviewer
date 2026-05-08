import torch
import os
import sys
import numpy as np
import evaluate
from PIL import Image
from datasets import Dataset, Features, ClassLabel, Image as DatasetImage
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)

# --- CONFIGURATION ---
# Path where your aircraft folders are located
DATA_PATH = r"D:\Development\AviationDataSet" 
MODEL_ID = "dima806/military_aircraft_image_detection"
OUTPUT_DIR = "./aviation_model_v2"

def train():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: DATA_PATH not found at {DATA_PATH}")
        print("Please check the path to your AviationDataSet folder.")
        return

    print(f"--- Scanning Dataset: {DATA_PATH} ---")
    
    # 1. Manually build the image list and labels to avoid Windows path issues
    image_paths = []
    labels = []
    # Get all subdirectories (class names)
    class_names = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    if not class_names:
        print("ERROR: No subdirectories found in DATA_PATH!")
        return

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
        print("ERROR: No images found in subdirectories!")
        return

    print(f"\nTotal images found: {len(image_paths)} across {len(class_names)} classes.")

    # 2. Create HuggingFace Dataset
    # Features definition ensures the 'image' column is treated as a path to an image file
    features = Features({
        "image": DatasetImage(),
        "label": ClassLabel(names=class_names)
    })
    
    raw_dataset = Dataset.from_dict({
        "image": image_paths,
        "label": labels
    }, features=features)

    # Split into 90% train, 10% validation
    print("--- Splitting Dataset (90% Train / 10% Eval) ---")
    dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    
    id2label = {str(i): label for i, label in enumerate(class_names)}
    label2id = {label: str(i) for i, label in enumerate(class_names)}

    # 3. Preprocessing
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    
    def transform(example_batch):
        # Convert all images to RGB (handles RGBA or Grayscale)
        images = [x.convert("RGB") for x in example_batch["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = example_batch["label"]
        return inputs

    # with_transform applies the preprocessing on-the-fly during training
    prepared_ds = dataset.with_transform(transform)

    # 4. Load Model with new head
    print(f"\n--- Initializing Model: {MODEL_ID} ---")
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # Replaces the 50-class head with our new one
    )

    # 5. Training Arguments
    # Optimized for a single GPU with 8GB+ VRAM. 
    # If you get Out of Memory, reduce per_device_train_batch_size to 8.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        fp16=torch.cuda.is_available(), # Uses GPU acceleration if available
    )

    # 6. Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    # 7. Run Training
    print("\n--- Starting Fine-Tuning ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    # 8. Save result
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    
    # Save the label list to a text file for RAWviewer update
    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(class_names))
        
    print(f"\nSUCCESS! Training complete. Best model saved to {OUTPUT_DIR}")
    print("Now run 'python src/export_to_onnx.py' to generate the RAWviewer model file.")

if __name__ == "__main__":
    train()
