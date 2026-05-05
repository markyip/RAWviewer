import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from semantic_search import resolve_mobileclip_backend

backend = resolve_mobileclip_backend()
print(f"Backend type: {type(backend).__name__}")
print(f"Available: {backend.available()}")
if not backend.available():
    print(f"Error: {backend.availability_error()}")
    
print(f"Model Dir: {backend.model_dir}")
print(f"Image Path: {backend.image_model_path}")
print(f"Text Path: {backend.text_model_path}")
print(f"Tokenizer Path: {backend.tokenizer_path}")

try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Providers: {ort.get_available_providers()}")
except ImportError:
    print("ONNX Runtime NOT installed")
