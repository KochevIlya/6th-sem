# Импорт необходимых библиотек
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation

# Функция для отображения изображений
def show_images(images, columns=2, figsize=(10, 10)):
    """
    Функция для визуализации изображений.
    :param images: список изображений или одно изображение
    :param columns: количество столбцов для отображения
    :param figsize: размеры фигуры
    """
    if isinstance(images, np.ndarray):
        images = [images]
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = np.array(axes).flatten()
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray', interpolation='none')
        axes[i].axis('off')
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Загрузка изображения
image_path = 'hourse.png'  # Укажите путь к вашему изображению
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Файл {image_path} не найден. Проверьте путь.")
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
binary_image = original_image > 127  # Преобразование в бинарное изображение

# Визуализация исходного изображения
show_images(binary_image, columns=1, figsize=(6, 6))

# Создание структурирующего элемента
radius = 3
structuring_element = np.array([
    [
        np.hypot(row, col) <= radius for col in range(-radius, radius + 1)
    ] for row in range(-radius, radius + 1)
], dtype=bool)

# Визуализация структурирующего элемента
show_images(structuring_element, columns=1, figsize=(4, 4))

# Построение морфологического скелета
def compute_skeleton(binary_img, struct_elem, return_all_steps=False):
    """
    Функция для построения морфологического скелета.
    :param binary_img: бинарное изображение
    :param struct_elem: структурирующий элемент
    :param return_all_steps: если True, возвращает все шаги
    :return: скелет изображения или список шагов
    """
    binary_img = binary_img.astype(bool)
    struct_elem = struct_elem.astype(bool)
    skeleton = np.zeros_like(binary_img, dtype=bool)
    steps = [] if return_all_steps else None

    while True:
        eroded = binary_erosion(binary_img, structure=struct_elem)
        if not eroded.any():
            break
        temp = binary_dilation(eroded, structure=struct_elem)
        diff = binary_img & ~temp
        skeleton |= diff
        if return_all_steps:
            steps.append(diff)
        binary_img = eroded

    if return_all_steps:
        steps.append(binary_img)
        return steps
    else:
        return skeleton

# Восстановление изображения по шагам
def reconstruct_image(steps, struct_elem):
    """
    Функция для восстановления изображения по шагам.
    :param steps: шаги построения скелета
    :param struct_elem: структурирующий элемент
    :return: восстановленное изображение
    """
    reconstructed = np.zeros_like(steps[0], dtype=bool)
    for step in reversed(steps):
        reconstructed = binary_dilation(reconstructed, struct_elem) | step
    return reconstructed

# Построение скелета
skeleton_steps = compute_skeleton(binary_image, structuring_element, return_all_steps=True)
skeleton_image = np.any(skeleton_steps, axis=0)

# Визуализация скелета
show_images(skeleton_image, columns=1, figsize=(6, 6))

# Восстановление изображения
reconstructed_image = reconstruct_image(skeleton_steps, structuring_element)

# Визуализация восстановленного изображения
show_images(reconstructed_image, columns=1, figsize=(6, 6))

# Проверка корректности восстановления
if np.array_equal(binary_image, reconstructed_image):
    print("Восстановление успешно!")
else:
    print("Ошибка восстановления!")