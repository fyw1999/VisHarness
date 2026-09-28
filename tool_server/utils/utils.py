import base64
import json
import threading
from io import BytesIO

from PIL import Image

_file_locks = {}
_lock_lock = threading.Lock()

def _get_file_lock(filepath):
    """Get a lock for a specific file path."""
    with _lock_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]

def load_json_file(filepath):
    '''
        Read a JSON file into a list or dictionary in a thread-safe manner.
    '''
    with _get_file_lock(filepath):
        with open(filepath, 'r', encoding="UTF-8") as file:
            data = json.load(file)
    return data

def write_json_file(data, filepath):
    '''
        Write a JSON file in a thread-safe manner.
    '''
    with _get_file_lock(filepath):
        with open(filepath, 'w', encoding="UTF-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def write_jsonl(data, file_path):
    '''
        Write a list of dictionaries to a JSONL file in a thread-safe manner.
    '''
    with _get_file_lock(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                line = json.dumps(item, ensure_ascii=False)
                file.write(line + '\n')

def bytes_to_pil(image_bytes):
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def pil_to_bytes(image_pil):
    img_byte_arr = BytesIO()
    image_pil.save(img_byte_arr, format='JPEG', quality=95, subsampling=0)
    img_bytes = img_byte_arr.getvalue()
    return img_bytes

def pil_to_base64(img: Image.Image, url_format = False) -> str:
    """
    Convert a PIL image to a base64 encoded string.
    
    Args:
        img (Image.Image): The PIL image to convert.
        
    Returns:
        str: Base64 encoded string representation of the image.
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    if url_format:
        img_str = f"data:image/jpeg;base64,{img_str}"
    return img_str
