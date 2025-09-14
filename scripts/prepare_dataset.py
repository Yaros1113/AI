import os
import cv2
import numpy as np
import shutil
from PIL import Image
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

class ZeroShotLabeling:
    def __init__(self, model_path='best.pt'):
        # Используем обученную модель для детекции
        self.yolo_model = YOLO(model_path)
        
        # Установим пороги уверенности
        self.confidence_threshold = 0.5  # Порог уверенности для детекции
        self.iou_threshold = 0.4         # для NMS

    def detect_logo_candidates(self, image_path):
        """Обнаружение логотипов с помощью ранее обученной модели YOLO"""
        results = self.yolo_model(image_path, conf=self.confidence_threshold, 
                                iou=self.iou_threshold, imgsz=640)
        candidates = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Координаты bbox
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = box.cls[0].cpu().numpy()
                    
                    # Проверяем, что это именно логотип Т-Банка (класс 0)
                    if class_id == 0 and confidence >= self.confidence_threshold:
                        candidates.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence
                        })
        
        return candidates

    def process_image(self, image_path, output_dir, negative_dir):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        height, width = image.shape[:2]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # Детектируем логотипы обученной моделью
        candidates = self.detect_logo_candidates(image_path)
        if not candidates:
            # Создаем пустой файл разметки для изображений без логотипов
            open(txt_path, 'w').close()
            shutil.copy(image_path, negative_dir)
            return False
        
        valid_detections = []
        for candidate in candidates:
            bbox = candidate['bbox']
            # Конвертируем bbox в формат YOLO (x_center, y_center, w, h) относительно размеров
            x_center = ((bbox[0] + bbox[2]) / 2) / width
            y_center = ((bbox[1] + bbox[3]) / 2) / height
            w = (bbox[2] - bbox[0]) / width
            h = (bbox[3] - bbox[1]) / height
            
            # Класс 0 для логотипа Т-Банка
            valid_detections.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        # Записываем результаты в файл
        with open(txt_path, 'w') as f:
            f.write("\n".join(valid_detections))
        
        # Копируем изображение в папку positive
        if valid_detections:
            shutil.copy(image_path, output_dir)
            return True
        else:
            shutil.copy(image_path, negative_dir)
            return False

    def process_directory(self, input_dir, output_dir, negative_dir):
        """Обрабатывает все изображения в директории"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(negative_dir):
            os.makedirs(negative_dir)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_paths = []
        
        for f in os.listdir(input_dir):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(input_dir, f))
        
        print(f"Found {len(image_paths)} images for processing")
        
        successful_processing = []
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            success = self.process_image(image_path, output_dir, negative_dir)
            if success:
                successful_processing.append(image_path)
        
        return successful_processing

    def create_dataset_config(self, output_dir):
        """Создает конфигурационный файл для датасета"""
        # Создаем YAML конфиг для датасета
        yaml_content = f"""path: {os.path.abspath(output_dir).replace('\\', '/')}
train: images/train
val: images/val
nc: 1
names: ['tbank_logo']
"""
        
        with open(os.path.join(output_dir, 'tbank_dataset.yaml'), 'w') as f:
            f.write(yaml_content)

def main():
    # Путь к обученной модели
    model_path = 'data/models/yolo11m_tbank_finetuned6/weights/best.pt'
    
    labeler = ZeroShotLabeling(model_path)
    
    raw_images_dir = 'data/input/raw'
    output_dir = 'data/input/positive'
    negative_dir  = 'data/input/positive/negative'
    
    # Обрабатываем изображения
    processed_images = labeler.process_directory(raw_images_dir, output_dir, negative_dir)
    
    if processed_images:
        # Создаем конфигурацию датасета
        labeler.create_dataset_config('data/input')
        print(f"Markup completed successfully! Processed {len(processed_images)} images with logos.")
    else:
        print("No logos found in any images")

if __name__ == "__main__":
    main()