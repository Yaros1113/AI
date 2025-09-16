import asyncio
import aiohttp
import time
from pathlib import Path
import random
import json
from aiohttp import FormData
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Any

def visualize_detections(image_path: Path, detections: List[Dict], output_dir: Path = None):
    """
    Визуализирует детекции на изображении
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    # Рисуем боксы для каждой детекции
    for detection in detections:
        bbox = detection.get('bbox', [])
        
        try:
            x1 = int(bbox.get('x_min', 0))
            y1 = int(bbox.get('y_min', 0))
            x2 = int(bbox.get('x_max', 0))
            y2 = int(bbox.get('y_max', 0))
            
            # Рисуем прямоугольник
            color = (0, 255, 0)  # Зеленый
            thickness = 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Рисуем фон для текста
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Рисуем текст
            cv2.putText(image, "Class 0", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        except (ValueError, TypeError) as e:
            print(f"Error parsing bbox {bbox}: {e}")
    
    # Конвертируем BGR в RGB для matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Показываем изображение
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Detections on {image_path.name}")
    plt.tight_layout()
    
    # Сохраняем если указана выходная директория
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detected_{image_path.name}"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        print(f"Saved visualization to: {output_path}")
    
    plt.show()

def get_content_type(image_path):
    """Определяем Content-Type в зависимости от расширения файла"""
    extension = image_path.suffix.lower()
    content_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    return content_types.get(extension, 'application/octet-stream')

async def test_single_image(session, url, image_path, save_detections: bool = False):
    """Тестирование одного изображения с возможностью сохранения детекций"""
    try:
        # Создаем FormData для корректной отправки файла
        data = FormData()
        content_type = get_content_type(image_path)
        
        data.add_field('file', 
                      open(image_path, 'rb'),
                      filename=image_path.name,
                      content_type=content_type)
        
        start_time = time.time()
        async with session.post(url, data=data) as response:
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # в миллисекундах
            
            if response.status == 200:
                response_data = await response.json()
                
                result = {
                    'image': image_path.name,
                    'status': response.status,
                    'response_time_ms': response_time,
                    'detections_count': len(response_data.get('detections', [])),
                    'detections': response_data.get('detections', []),
                    'success': True,
                    'content_type': content_type
                }
                
                # Сохраняем детекции если нужно
                if save_detections and result['detections_count'] > 0:
                    output_dir = Path("data/output/visualizations")
                    visualize_detections(image_path, result['detections'], output_dir)
                
                return result
            else:
                error_text = await response.text()
                return {
                    'image': image_path.name,
                    'status': response.status,
                    'response_time_ms': response_time,
                    'error': f"HTTP {response.status}: {error_text}",
                    'success': False,
                    'content_type': content_type
                }
                
    except aiohttp.ClientError as e:
        return {
            'image': image_path.name,
            'status': 0,
            'response_time_ms': 0,
            'error': f"Client error: {str(e)}",
            'success': False
        }
    except Exception as e:
        return {
            'image': image_path.name,
            'status': 0,
            'response_time_ms': 0,
            'error': f"Unexpected error: {str(e)}",
            'success': False
        }

async def load_test(api_url, image_dir, num_requests=10, concurrent_workers=5, visualize: bool = False):
    """Проведение нагрузочного тестирования с опцией визуализации"""
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        print(f"Error: Directory '{image_dir}' does not exist!")
        return
    
    # Получаем все изображения поддерживаемых форматов
    image_paths = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    for ext in supported_extensions:
        image_paths.extend(list(image_dir_path.glob(f'*{ext}')))
        image_paths.extend(list(image_dir_path.glob(f'*{ext.upper()}')))  # На случай заглавных расширений
    
    if not image_paths:
        print(f"No images found in '{image_dir}' with supported formats: {supported_extensions}")
        print("Please add images with extensions: .jpg, .jpeg, .png, .bmp, .webp")
        return
    
    # Выбираем случайные изображения для теста
    selected_images = random.sample(image_paths, min(num_requests, len(image_paths)))
    
    print(f"Starting load test with {len(selected_images)} images, {concurrent_workers} concurrent workers")
    print(f"API URL: {api_url}")
    print(f"Visualization: {'ENABLED' if visualize else 'DISABLED'}")
    print(f"Supported formats: {supported_extensions}")
    
    # Группируем изображения по формату для информации
    format_stats = {}
    for img in selected_images:
        ext = img.suffix.lower()
        format_stats[ext] = format_stats.get(ext, 0) + 1
    
    print(f"Image format distribution: {format_stats}")
    
    results = []
    start_time = time.time()
    
    # Увеличиваем таймаут для запросов
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Создаем все задачи сразу
        tasks = [test_single_image(session, api_url, img_path, visualize) for img_path in selected_images]
        
        # Выполняем задачи батчами
        for i in range(0, len(tasks), concurrent_workers):
            batch = tasks[i:i + concurrent_workers]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
            # Выводим прогресс
            print(f"Processed {len(results)}/{len(tasks)} requests")
            
            # Небольшая пауза между батчами
            if i + concurrent_workers < len(tasks):
                await asyncio.sleep(0.5)
    
    total_time = (time.time() - start_time) * 1000  # в миллисекундах
    
    # Анализ результатов
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"LOAD TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed requests details:")
        for fail in failed_tests[:5]:  # Показываем первые 5 ошибок
            print(f"  - {fail['image']}: {fail.get('error', 'Unknown error')}")
        if len(failed_tests) > 5:
            print(f"  ... and {len(failed_tests) - 5} more failures")
    
    if successful_tests:
        response_times = [r['response_time_ms'] for r in successful_tests]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        total_detections = sum(r['detections_count'] for r in successful_tests)
        
        # Анализ по форматам
        format_performance = {}
        for result in successful_tests:
            ext = Path(result['image']).suffix.lower()
            if ext not in format_performance:
                format_performance[ext] = {
                    'count': 0,
                    'total_time': 0,
                    'detections': 0
                }
            format_performance[ext]['count'] += 1
            format_performance[ext]['total_time'] += result['response_time_ms']
            format_performance[ext]['detections'] += result['detections_count']
        
        print(f"\nPerformance metrics:")
        print(f"Total time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        print(f"Average response time: {avg_response_time:.2f} ms")
        print(f"Min response time: {min_response_time:.2f} ms")
        print(f"Max response time: {max_response_time:.2f} ms")
        print(f"Total detections: {total_detections}")
        print(f"Requests per second: {len(successful_tests) / (total_time / 1000):.2f}")
        
        print(f"\nPerformance by format:")
        for ext, stats in format_performance.items():
            avg_time = stats['total_time'] / stats['count']
            avg_detections = stats['detections'] / stats['count']
            print(f"  {ext}: {stats['count']} requests, "
                  f"avg: {avg_time:.2f} ms, "
                  f"avg detections: {avg_detections:.1f}")
        
        # Проверяем ограничение в 10 секунд
        if max_response_time > 10000:
            print(f"❌ WARNING: Maximum response time ({max_response_time/1000:.2f}s) exceeds 10 second limit!")
        else:
            print(f"✅ SUCCESS: All responses under 10 seconds (max: {max_response_time/1000:.2f}s)")
    else:
        print("No successful requests to calculate performance metrics")
    
    return results

async def test_connection(api_url):
    """Проверка соединения с API"""
    try:
        async with aiohttp.ClientSession() as session:
            # Пробуем endpoint здоровья или просто основной URL
            health_url = api_url.replace('/detect', '/health')
            async with session.get(health_url, timeout=5) as response:
                if response.status == 200:
                    print("✅ API health check passed")
                    return True
                else:
                    print(f"❌ API health check failed: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please make sure the API server is running on http://localhost:8080")
        return False

def visualize_sample_detections(results: List[Dict], image_dir: Path, num_samples: int = 3):
    """
    Визуализирует детекции для нескольких случайных успешных запросов
    """
    successful_with_detections = [r for r in results if r['success'] and r['detections_count'] > 0]
    
    if not successful_with_detections:
        print("No successful detections to visualize")
        return
    
    # Выбираем случайные примеры для визуализации
    samples = successful_with_detections
    
    print(f"\n{'='*50}")
    print(f"VISUALIZING {len(samples)} SAMPLE DETECTIONS")
    print(f"{'='*50}")
    
    for sample in samples:
        image_path = image_dir / sample['image']
        if image_path.exists():
            print(f"Visualizing detections for: {sample['image']}")
            visualize_detections(image_path, sample['detections'])
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    API_URL = "http://localhost:8080/api/v1/detect"
    IMAGE_DIR = "data/input/test"
    VISUALIZE = False  # Включить визуализацию детекций
    SAMPLE_VISUALIZATIONS = 10  # Количество примеров для визуализации после теста
    
    # Проверяем содержимое директории
    image_dir = Path(IMAGE_DIR)
    if image_dir.exists():
        print(f"Contents of '{IMAGE_DIR}':")
        for item in image_dir.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                print(f"  - {item.name}")
    else:
        print(f"Directory '{IMAGE_DIR}' does not exist. Creating it...")
        image_dir.mkdir(parents=True, exist_ok=True)
        print("Please add some test images to the directory and run the test again.")
        exit(1)
    
    # Проверяем соединение перед тестом
    if asyncio.run(test_connection(API_URL)):
        # Запускаем тест
        results = asyncio.run(load_test(
            api_url=API_URL,
            image_dir=IMAGE_DIR,
            num_requests=50,
            concurrent_workers=3,
            visualize=VISUALIZE
        ))
        
        # Сохраняем результаты в файл
        with open('load_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to load_test_results.json")
        
        # Визуализируем несколько примеров после теста
        if not VISUALIZE:  # Если не визуализировали во время теста
            visualize_sample_detections(results, image_dir, SAMPLE_VISUALIZATIONS)
            
    else:
        print("Please make sure the API server is running on http://localhost:8080")