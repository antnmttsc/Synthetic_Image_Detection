# This Code regards the manipultaion and preprocessing of data without the S&R preprocess

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import cv2
import math

def set_seed(seed, use_gpu = True):
    """
    Set SEED for PyTorch reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def stratified_sample(dataset, percentage=0.1):
  """
  Perform stratified sampling where the number of samples for each class is kept proportional
  but at the same time we ensure the number of real images to be more or less the same of the fake ones
  """
  labels = dataset["label"]
  classes = set(labels)
  selected_indices = []
  
  for class_x in classes:
    num_samples_class = [i for i in range(len(labels)) if labels[i] == class_x]
    random.shuffle(num_samples_class)
    if class_x == 0:
      num_samples_class = num_samples_class[:int(len(num_samples_class)*percentage)]
    else:
      num_samples_class = num_samples_class[:int(len(num_samples_class)*percentage/(len(classes)-1))]
    selected_indices.extend(num_samples_class)
  
  stratified_subset = dataset.select(selected_indices)
  return stratified_subset


def plot_exemples(
    data,
    lbl_occ,
    generation_methods:list=['authentic', 'dalle-3-images', 'diffusiondb', 'midjourney-images', 'midjourney_tti', 'realisticSDXL'],
):
    '''
    This function print a image for each of the generation methods in the data
    '''

    lbl_occ_cumsum = np.cumsum(lbl_occ)
    lbl_occ_cumsum = [l-1 for l in lbl_occ_cumsum]
    
    # this adapt dinamically the number of rows and cols for the plot
    num_methods = len(generation_methods)
    cols = min(3, num_methods)
    rows = math.ceil(num_methods / cols)  # compute necessary rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    for i, idx in enumerate(lbl_occ_cumsum):

        image = data[int(idx)]['image']
        label = data[int(idx)]['label']

        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"Class {label}: {generation_methods[label]}")

    # hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def resize_images(dataset, target_size=(256, 256), name:str=""):
  '''
  This function resize images with bicubic interpolation to ensure they have the same size
  '''

  resized_images = [(None, None)] * len(dataset)

  for i, item in tqdm(enumerate(dataset), total=len(dataset), desc=f"Resizing {name} Images"): # We used tqdm to check the progresses 
   
    image_np = np.array(item['image'])
    resized_image = cv2.resize(image_np, target_size, interpolation=cv2.INTER_CUBIC) # resize image
    resized_images[i] = (resized_image, item['label'])

  return resized_images


def MinMaxScaler(img):
  '''
  This function scale an image using the min-max scaler
  '''
  img_as_array = np.asarray(img).astype(np.float32) 
  arr_to_tensor = torch.tensor(img_as_array) 
  tensor_permuted = arr_to_tensor.permute(2, 0, 1)  # from HWC to CHW
  normalized_img = tensor_permuted.float() / 255

  return normalized_img


def check_class_distribution(dataloader):
  '''
  This function simply count the number of occurrences for each class in the dataloader
  '''
  class_counts = {}
  for _, targets in dataloader:
    for label in targets:
      label = label.item()
      if label not in class_counts:
        class_counts[label] = 1
      else:
        class_counts[label] += 1

  sorted_class_counts = dict(sorted(class_counts.items()))

  return sorted_class_counts






















