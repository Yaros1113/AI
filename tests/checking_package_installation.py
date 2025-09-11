import torch
import cv2
from fastapi import FastAPI
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import subprocess

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV version:", cv2.__version__)
print("FastAPI imported successfully")
input()