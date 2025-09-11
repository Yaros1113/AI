import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import subprocess

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("T-Bank Logo Detection Control Panel")
        self.geometry("700x500")

        # Кнопки
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        self.btn_labeling = tk.Button(btn_frame, text="Разметка данных", width=20, command=self.run_labeling)
        self.btn_labeling.grid(row=0, column=0, padx=5)

        self.btn_training = tk.Button(btn_frame, text="Обучение модели", width=20, command=self.run_training)
        self.btn_training.grid(row=0, column=1, padx=5)

        self.btn_ml_service = tk.Button(btn_frame, text="Запуск ML-сервиса", width=20, command=self.run_ml_service)
        self.btn_ml_service.grid(row=0, column=2, padx=5)

        self.btn_validation = tk.Button(btn_frame, text="Валидация модели", width=20, command=self.run_validation)
        self.btn_validation.grid(row=0, column=3, padx=5)

        # Текстовое поле для логов
        self.log_area = scrolledtext.ScrolledText(self, height=25, state='disabled')
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def log(self, message):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

    def run_in_thread(self, target):
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def run_command(self, command, description):
        self.log(f"Запуск: {description}...")
        try:
            process = subprocess.Popen(
                command,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                shell=True
            )
            for line in process.stdout:
                self.log(line.strip())
            process.wait()
            if process.returncode == 0:
                self.log(f"{description} завершено успешно.\n")
            else:
                self.log(f"Ошибка при выполнении {description}. Код: {process.returncode}\n")
                messagebox.showerror("Ошибка", f"{description} завершилось с ошибкой.")
        except Exception as e:
            self.log(f"Исключение: {e}")
            messagebox.showerror("Ошибка", str(e))

    # Обработчики кнопок (запуск в потоках) ??? Добавить сборку докера
    def run_labeling(self):
        # Пример запуска скрипта разметки
        self.run_in_thread(lambda: self.run_command("python scripts/prepare_dataset.py", "Разметка данных"))

    def run_training(self):
        # Пример запуска обучения модели
        self.run_in_thread(lambda: self.run_command("python scripts/train.py", "Обучение модели"))

    def run_ml_service(self):
        # Запуск FastAPI сервиса через докер
        self.run_in_thread(lambda: self.run_command("run_docker.bat", "ML-сервис"))

    def run_validation(self):
        # Запуск скрипта валидации
        self.run_in_thread(lambda: self.run_command("python src/scripts/validate.py", "Валидация модели"))

if __name__ == "__main__":
    app = App()
    app.mainloop()