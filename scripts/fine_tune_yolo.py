import os
from ultralytics import YOLO
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset_structure(raw_images_dir, base_output_dir):
    # Пути для YOLO
    images_train_dir = os.path.join(base_output_dir, 'images', 'train')
    images_val_dir = os.path.join(base_output_dir, 'images', 'val')
    labels_train_dir = os.path.join(base_output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(base_output_dir, 'labels', 'val')

    # Создаем директории
    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Собираем все изображения (предполагаем, что разметка уже есть)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    all_images = []
    for file in os.listdir(raw_images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            all_images.append(file)

    # Разделяем на train и val
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

    # Функция для копирования файлов
    def copy_files(file_list, images_dest_dir, labels_dest_dir):
        for i, img_file in enumerate(file_list):
            print(f"Processing {i+1} image: {img_file}")
            # Копируем изображение
            src_img = raw_images_dir+"\\"+img_file
            dst_img = images_dest_dir +"\\"+ img_file
            print(f"src_img: {src_img}, dst_img: {dst_img}")
            shutil.copy2(src_img, dst_img)

            # Копируем соответствующий файл разметки (.txt)
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            src_txt = raw_images_dir +"\\"+ txt_file
            if os.path.exists(src_txt):
                dst_txt = labels_dest_dir +"\\"+ txt_file
                shutil.copy2(src_txt, dst_txt)
            else:
                # Если разметки нет, создаем пустой файл
                open(os.path.join(labels_dest_dir, txt_file), 'w', encoding='utf-8').close()

    # Копируем файлы в соответствующие директории
    copy_files(train_images, images_train_dir, labels_train_dir)
    copy_files(val_images, images_val_dir, labels_val_dir)

    print(f"Dataset prepared. Train: {len(train_images)} images, Val: {len(val_images)} images.")
    return len(train_images), len(val_images)

def fine_tune_yolo():
    """
    Основная функция для тонкой настройки YOLO на датасете логотипов Т-Банка.
    """
    # Пути
    raw_data_dir = 'data/input/positive'  # Ваша папка с сырыми изображениями и разметкой
    prepared_data_dir = 'data/input/'  # Куда сохранить подготовленные данные
    dataset_config_path = 'data/input/tbank_dataset.yaml'
    model_save_dir = 'data/models'

    # 1. Подготовка структуры датасета
    prepare_dataset_structure(raw_data_dir, prepared_data_dir)

    # 2. Загружаем предобученную модель YOLO
    # Рекомендуется использовать 'yolo11m.pt' как баланс скорости и точности
    model = YOLO('data/models/yolo11m_tbank_finetuned6/weights/best.pt')

    # 3. Запускаем обучение (fine-tuning)
    results = model.train(
        data=dataset_config_path,   # Путь к YAML-конфигу
        epochs=50,                  # Количество эпох
        imgsz=640,                  # Размер изображения
        batch=16,                   # Размер батча (зависит от памяти GPU)
        patience=15,                # Ранняя остановка, если нет улучшений
        project=model_save_dir,     # Куда сохранять результаты
        name='yolo11m_tbank_finetuned', # Имя эксперимента
        optimizer='AdamW',          # Оптимизатор
        lr0=0.001,                  # Начальная скорость обучения
        # cos_lr=True,              # Использовать косинусный расписание LR (опционально)
        # weight_decay=0.0005,      # L2 регуляризация (опционально)
        # augment=True,             # Аугментация данных (включена по умолчанию)
        # freeze=10,                 # Заморозить первые 10 слоев (опционально, для экономии памяти и ускорения, если датасет маленький)
        device='0',                 # Использовать GPU с ID 0
        verbose=True                # Выводить подробный прогресс
    )

    # 4. Оцениваем модель на валидационном наборе
    #best_model_path = results.best
    best_model = YOLO("data/models/yolo11m_tbank_best.pt")#best_model_path)
    metrics = best_model.val(
        device='0',
    )  # Этот метод вернет объект с метриками

    # Печатаем метрики через встроенные атрибуты
    print("\n=== Validation Results ===")
    print(f"Precision (mp): {metrics.box.mp}")
    print(f"Recall (mr): {metrics.box.mr}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"F1-score: {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr):.3f}")

    # Raw-данные
    print(f"Raw Precision list (per class): {metrics.box.p}")
    print(f"Raw Recall list (per class): {metrics.box.r}")

    # 5. Экспорт модели
    #model.export(format='onnx', device='0')   # Для ускорения инференса

if __name__ == "__main__":
    fine_tune_yolo()