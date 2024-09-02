import torch
import io

from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import PIL.ImageOps


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(16 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)  # x = x.view(-1, 16 * 22 * 22)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = torch.load("./models/MnistModel.pth", weights_only=False)
model.eval()


def predict_digit(img):
    # изменение размера изобржений на 28x28
    transform = transforms.Compose([
        transforms.Resize((26, 26)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # конвертируем rgb в grayscale
    img = img.convert('L')

    plt.imshow(img, cmap="grey")
    plt.show()
    # Примените трансформации
    image_tensor = transform(img).unsqueeze(0)  # Добавьте размерность батча

    with torch.no_grad():
        output = model(image_tensor)

    # Получите вероятности и класс
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    predicted_probability = probabilities[0][predicted_class].item()

    print(f"Predicted class: {predicted_class}")
    print(f"Probability of the predicted class: {predicted_probability:.4f}")

    return predicted_class, predicted_probability


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=300, bg="black", cursor="draped_box")
        self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 24), bg="black", fg="white")
        self.classify_btn = tk.Button(self, text="Распознать", command=self.classify_handwriting, bg="black", fg="white")
        self.button_clear = tk.Button(self, text="Очистить", command=self.clear_all, bg="black", fg="white")
        self.button_info = tk.Button(self, text="Info", command=self.print_info, bg="black", fg="white")

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.button_info.grid(row=1, column=2, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):

        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab((rect[0] + 23, rect[1] + 23, rect[2] + 80, rect[3] + 80))

        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='white', outline="")

    def print_info(self):
        print(win32gui.GetWindowRect(self.canvas.winfo_id()))
        print(self.x, self.y)


app = App()
app["bg"] = "black"
mainloop()