import os
import sys
import csv
import torch
import numpy as np
import rawpy
from PIL import Image
from tqdm import tqdm
from skimage.measure import label, regionprops
from rembg import remove, new_session
from transformers import ViTForImageClassification, ViTImageProcessor

# Add src path to import exif_subject_area
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from exif_subject_area import pixmap_ltwh_focus_hint

MODEL_DIR = "./aviation_model_processed"
INPUT_DIR = r"D:\Development\Test Image set"
OUTPUT_DIR = r"D:\Development\Test_Images_Pipeline_Results"

MIN_WIDTH = 350
MIN_HEIGHT = 350

def get_focus_point(filepath, width, height):
    """Returns (cx, cy) focus point or None."""
    hint = pixmap_ltwh_focus_hint(filepath, width, height)
    if not hint:
        return None
    ltwh, _ = hint
    l, t, w, h = ltwh
    return l + w / 2.0, t + h / 2.0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading identifier model from {MODEL_DIR}...")
    try:
        processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
        model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Initializing rembg session with isnet-general-use...")
    session = new_session("isnet-general-use")

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.arw') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    print(f"Found {len(files)} files to process in {INPUT_DIR}.")

    csv_path = os.path.join(OUTPUT_DIR, "classification_results.csv")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Status", "Rank 1 Label", "Rank 1 Conf", "Rank 2 Label", "Rank 2 Conf", "Rank 3 Label", "Rank 3 Conf", "Error"])
        
        for filename in tqdm(files, desc="Processing Images"):
            filepath = os.path.join(INPUT_DIR, filename)
            out_img_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_trimmed.png")
            
            try:
                # 1. Load Image
                if filepath.lower().endswith('.arw'):
                    with rawpy.imread(filepath) as raw:
                        rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    img = Image.fromarray(rgb)
                else:
                    img = Image.open(filepath).convert("RGB")
                
                # Get focus point
                focus_point = get_focus_point(filepath, img.width, img.height)
                
                # 2. Remove BG (High Accuracy)
                img_nobg = remove(img, session=session, alpha_matting=False)
                
                # 3. Connected Components on Alpha Mask
                alpha = np.array(img_nobg.split()[-1])
                binary_mask = alpha > 20
                
                labeled_mask = label(binary_mask)
                props = regionprops(labeled_mask)
                
                if not props:
                    writer.writerow([filename, "Empty Mask", "", "", "", "", "", "", "No object detected after background removal"])
                    f.flush()
                    continue
                
                target_blob_label = None
                
                # Find blob overlapping with focus point
                if focus_point:
                    cx, cy = focus_point
                    for p in props:
                        minr, minc, maxr, maxc = p.bbox
                        if minc <= cx <= maxc and minr <= cy <= maxr:
                            target_blob_label = p.label
                            break
                            
                # Fallback: keep the largest blob
                if target_blob_label is None:
                    target_blob_label = max(props, key=lambda p: p.area).label
                
                # 4. Isolate the chosen blob
                blob_mask = (labeled_mask == target_blob_label)
                # Apply mask to original alpha
                new_alpha = np.where(blob_mask, alpha, 0).astype(np.uint8)
                
                # Create final isolated image
                final_img = img_nobg.copy()
                final_img.putalpha(Image.fromarray(new_alpha))
                
                # 5. Trim Empty Space using isolated alpha
                bbox = Image.fromarray(new_alpha).getbbox()
                if not bbox:
                    writer.writerow([filename, "Empty Blob", "", "", "", "", "", "", ""])
                    f.flush()
                    continue
                    
                cropped_img = final_img.crop(bbox)
                width, height = cropped_img.size
                
                # Save the trimmed image
                cropped_img.save(out_img_path)
                
                # 6. Small Object Filtering
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    writer.writerow([filename, f"Object Too Small ({width}x{height})", "", "", "", "", "", "", "Filtered out"])
                    f.flush()
                    continue
                
                # 7. Model Inference
                background = Image.new("RGB", cropped_img.size, (255, 255, 255))
                background.paste(cropped_img, mask=cropped_img.split()[3])
                
                inputs = processor(images=background, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                top3_prob, top3_indices = torch.topk(probs, 3)
                
                status_msg = "Success (Focused Blob)" if focus_point else "Success (Largest Blob)"
                row = [filename, status_msg]
                for i in range(3):
                    idx = top3_indices[i].item()
                    label_name = model.config.id2label[idx]
                    conf = top3_prob[i].item() * 100
                    row.extend([label_name, f"{conf:.2f}%"])
                row.append("") # No error
                
            except Exception as e:
                row = [filename, "Error", "", "", "", "", "", "", str(e)]
            
            writer.writerow(row)
            f.flush()

if __name__ == "__main__":
    main()
