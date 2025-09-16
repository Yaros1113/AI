from io import BytesIO
from PIL import Image
import numpy as np

def validate_image(image_bytes: bytes) -> None:
    """Валидация, что bytes являются корректным изображением"""
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()  # Проверяет целостность файла
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

async def process_image(image_bytes: bytes, target_size: int = 640) -> np.ndarray:
    """
    Конвертирует bytes изображения в PIL Image для YOLO с ресайзом
    Args:
        image_bytes: Изображение в виде байтов
        target_size: Целевой размер для ресайза (по наибольшей стороне)
    Returns:
        PIL.Image: Изображение в формате PIL Image
    """
    
    # Открываем и конвертируем в RGB
    image = Image.open(BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Ресайз с сохранением пропорций
    original_width, original_height = image.size
    scale_factor = target_size / max(original_width, original_height)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image