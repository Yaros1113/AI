import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split

class ZeroShotLabeling:
    def __init__(self):
        self.yolo_model = YOLO('yolov8m.pt')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Определяем промпты
        self.positive_prompts = [
            "a black T on a yellow shield logo",
            "T-Bank logo", 
            "a white T on a grey shield",
            "a black T on a white shield",
            "a black T on a silver shield",
            "a geometric T emblem",
            "stylized letter T in a shield"
        ]
        self.negative_prompts = [
            "Tinkoff logo",
            "a red T on a circle",
            "text",
            "a person",
            "a car",
            "background",
            "object"
        ]
        
        # Пороговые значения
        self.confidence_threshold = 0.5  # для YOLO
        self.clip_positive_threshold = 0.7  # для CLIP
        self.clip_negative_threshold = 0.3  # для CLIP

    def detect_logo_candidates(self, image_path):
        """Обнаружение кандидатов в логотипы с помощью YOLOv8"""
        results = self.yolo_model(image_path, conf=self.confidence_threshold)
        candidates = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Получаем координаты bbox
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    candidates.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
        
        return candidates

    def crop_bbox(self, image, bbox):
        """Вырезание области из изображения"""
        x1, y1, x2, y2 = map(int, bbox)
        return image[y1:y2, x1:x2]

    def clip_classify(self, image_patch):
        """Классификация патча с помощью CLIP"""
        # Преобразуем в PIL Image
        patch_pil = Image.fromarray(cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB))
        
        # Подготавливаем текстовые промпты
        texts = self.positive_prompts + self.negative_prompts
        
        # Обрабатываем изображение и тексты
        inputs = self.clip_processor(
            text=texts, 
            images=patch_pil, 
            return_tensors="pt", 
            padding=True
        )
        
        # Получаем предсказания
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Разделяем вероятности на положительные и отрицательные
        positive_probs = probs[:len(self.positive_prompts)]
        negative_probs = probs[len(self.positive_prompts):]
        
        return positive_probs, negative_probs

    def process_image(self, image_path, output_dir):
        """Обработка одного изображения"""
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        height, width = image.shape[:2]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # Обнаруживаем кандидатов
        candidates = self.detect_logo_candidates(image_path)
        valid_detections = []
        
        # Обрабатываем каждого кандидата
        for candidate in candidates:
            bbox = candidate['bbox']
            
            # Вырезаем патч
            patch = self.crop_bbox(image, bbox)
            if patch.size == 0:
                continue
            
            # Классифицируем с помощью CLIP
            positive_probs, negative_probs = self.clip_classify(patch)
            
            # Проверяем условия
            max_positive_prob = np.max(positive_probs)
            max_negative_prob = np.max(negative_probs)
            
            if (max_positive_prob > self.clip_positive_threshold and 
                max_positive_prob > max_negative_prob):
                # Преобразуем bbox в YOLO формат
                x_center = ((bbox[0] + bbox[2]) / 2) / width
                y_center = ((bbox[1] + bbox[3]) / 2) / height
                w = (bbox[2] - bbox[0]) / width
                h = (bbox[3] - bbox[1]) / height
                
                valid_detections.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        # Сохраняем результаты
        if valid_detections:
            with open(txt_path, 'w') as f:
                f.write("\n".join(valid_detections))
            return True
        else:
            # Создаем пустой файл, если нет обнаружений
            open(txt_path, 'w').close()
            return False

    def process_directory(self, input_dir, output_dir):
        """Обработка всех изображений в директории"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        # Собираем пути к изображениям
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(input_dir, file))
        
        print(f"Found {len(image_paths)} images for processing")
        
        # Обрабатываем каждое изображение
        successful_processing = []
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            success = self.process_image(image_path, output_dir)
            if success:
                successful_processing.append(image_path)
        
        return successful_processing

    def split_dataset(self, image_paths, output_dir, test_size=0.2):
        """Разделение данных на обучающую и валидационную выборки"""
        # Разделяем изображения
        train_paths, val_paths = train_test_split(
            image_paths, test_size=test_size, random_state=42
        )
        
        # Сохраняем пути для train и val
        with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
            f.write("\n".join(train_paths))
        
        with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
            f.write("\n".join(val_paths))
        
        return train_paths, val_paths

def main():
    # Инициализация
    labeler = ZeroShotLabeling()
    
    # Пути
    raw_images_dir = 'data/input/raw'
    output_labels_dir = 'data/input/labels'
    
    # Обрабатываем все изображения
    processed_images = labeler.process_directory(raw_images_dir, output_labels_dir)
    
    if processed_images:
        # Создаем файлы train.txt и val.txt
        labeler.split_dataset(
            processed_images, 
            'data/input',
            test_size=0.2
        )
        print("Markup completed successfully!")
    else:
        print("Failed to process any images")

if __name__ == "__main__":
    main()







'''
all_images = os.listdir('data/input/positive/') + os.listdir('data/input/negative/')
random.shuffle(all_images)
train_size = int(0.8 * len(all_images))

for i, img in enumerate(all_images):
    src = f'data/input/positive/{img}' if img in os.listdir('data/input/positive/') else f'data/input/negative/{img}'
    dest = 'data/train/' if i < train_size else 'data/val/'
    shutil.copy(src, dest)
    '''