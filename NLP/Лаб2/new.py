import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import binary_erosion

def show_images(images, columns=2, figsize=(10, 10)):
    
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

image_path = 'leaf.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
binary_image = original_image > 127 

show_images(binary_image, columns=1, figsize=(6, 6))

base_pattern = np.array([1] * 3 + [0] * 5, dtype=bool)
rotation_order = [0, 1, 3, 7, 8, 2, 6, 4, 5]

miss_structures = np.array([
    np.roll(np.concatenate((np.roll(base_pattern, shift), [False])), shift)[rotation_order].reshape((3, 3))
    for shift in range(8)
])
hit_structures = np.roll(miss_structures, 4, axis=0)
hit_structures[:, 1, 1] = 1
hit_structures[1::2, ::2, ::2] = 0

def apply_hit_and_miss(image, hit_structure, miss_structure):
    hit_result = binary_erosion(image, hit_structure)
    miss_result = binary_erosion(~image, miss_structure)
    return image & ~(hit_result & miss_result)

def perform_thinning_step(image):
    for hit_elem, miss_elem in zip(hit_structures, miss_structures):
        image = apply_hit_and_miss(image, hit_elem, miss_elem)
    return image

def generate_skeleton(image):
    prev_image = np.zeros_like(image)
    while not np.array_equal(image, prev_image): 
        prev_image = image.copy()
        image = perform_thinning_step(image)
    return image


skeleton = skeletonize(binary_image)
show_images(skeleton)