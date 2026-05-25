import os
import json
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Hugging Face training history")
    parser.add_argument("--model-dir", type=str, default="./aviation_model_processed", 
                        help="Path to the trained model directory containing trainer_state.json")
    args = parser.parse_args()

    state_file = os.path.join(args.model_dir, "trainer_state.json")
    
    if not os.path.exists(state_file):
        print(f"Error: Could not find {state_file}")
        print("Make sure you are pointing to a completed Hugging Face Trainer output directory.")
        return

    with open(state_file, "r") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    
    if not log_history:
        print("No training logs found in trainer_state.json")
        return

    train_epochs = []
    train_loss = []
    
    val_epochs = []
    val_loss = []
    val_accuracy = []

    for log in log_history:
        if "loss" in log and "epoch" in log:
            train_epochs.append(log["epoch"])
            train_loss.append(log["loss"])
        
        if "eval_loss" in log and "epoch" in log:
            val_epochs.append(log["epoch"])
            val_loss.append(log["eval_loss"])
            
        if "eval_accuracy" in log and "epoch" in log:
            val_accuracy.append(log["eval_accuracy"])

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss Plot
    if train_loss:
        ax1.plot(train_epochs, train_loss, label="Training Loss", color="blue", alpha=0.6)
    if val_loss:
        ax1.plot(val_epochs, val_loss, label="Validation Loss", color="red", marker='o')
    
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Accuracy Plot
    if val_accuracy:
        ax2.plot(val_epochs, val_accuracy, label="Validation Accuracy", color="green", marker='o')
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.7)
    else:
        ax2.text(0.5, 0.5, 'No accuracy metrics found', horizontalalignment='center', verticalalignment='center')
        ax2.set_title("Validation Accuracy")

    plt.tight_layout()
    
    save_path = os.path.join(args.model_dir, "training_history.png")
    plt.savefig(save_path)
    print(f"Plot saved successfully to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
