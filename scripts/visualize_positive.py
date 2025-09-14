import os
import cv2
import matplotlib.pyplot as plt

def yolo_to_bbox(yolo_bbox, img_width, img_height):
    """
    Конвертирует bbox из формата YOLO (x_center, y_center, w, h) в (x1, y1, x2, y2)
    """
    x_center, y_center, w, h = map(float, yolo_bbox)
    x1 = int((x_center - w / 2) * img_width)
    y1 = int((y_center - h / 2) * img_height)
    x2 = int((x_center + w / 2) * img_width)
    y2 = int((y_center + h / 2) * img_height)
    return x1, y1, x2, y2

def visualize_bboxes(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    img_height, img_width = image.shape[:2]

    # Читаем bbox из файла
    if not os.path.exists(label_path):
        print(f"Annotation file not found: {label_path}")
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Рисуем боксы
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, w, h = parts
        x1, y1, x2, y2 = yolo_to_bbox([x_center, y_center, w, h], img_width, img_height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Class {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Показываем изображение с bbox
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(os.path.basename(image_path))
    plt.show()

def main():
    positive_dir = 'data/input/positive'

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in os.listdir(positive_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]

    if not images:
        print("No positive images found.")
        return

    i = 0
    for img_file in images:
        i+=1
        if i < 100:
            image_path = os.path.join(positive_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(positive_dir, label_file)
            print(f"Preview: {img_file}")
            visualize_bboxes(image_path, label_path)
        else:
            break
    input()

if __name__ == "__main__":
    main()