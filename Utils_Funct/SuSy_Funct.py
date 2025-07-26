
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

def compute_top_n_patches(image, n_patches=5, patch_size=224):
  # Get the image dimensions
  width, height = image.size

  # Calculate the number of patches
  num_patches_x = width // patch_size
  num_patches_y = height // patch_size

  if num_patches_x == 0 or num_patches_y == 0:
    print(f"The patch size cannot be larger than the image size! Image size ({height}, {width}) - Patch size: ({patch_size}, {patch_size})")

  # Divide the image in patches
  patches = np.zeros((num_patches_x * num_patches_y, patch_size, patch_size, 3), dtype=np.uint8)
  for i in range(num_patches_x):
      for j in range(num_patches_y):
          x = i * patch_size
          y = j * patch_size
          patch = image.crop((x, y, x + patch_size, y + patch_size))
          patches[i * num_patches_y + j] = np.array(patch)

  # Compute the most relevant patches (optional)
  dissimilarity_scores = []
  for patch in patches:
      transform_patch = transforms.Compose([transforms.PILToTensor(), transforms.Grayscale()])
      grayscale_patch = transform_patch(Image.fromarray(patch)).squeeze(0)
      glcm = graycomatrix(grayscale_patch, [5], [0], 256, symmetric=True, normed=True)
      dissimilarity_scores.append(graycoprops(glcm, "contrast")[0, 0])

  # Sort patch indices by their dissimilarity score
  sorted_indices = np.argsort(dissimilarity_scores)[::-1]

  # Extract top k patches and convert them to tensor
  top_patches = patches[sorted_indices[:n_patches]]
  top_patches = torch.from_numpy(np.transpose(top_patches, (0, 3, 1, 2))) / 255.0
  
  return top_patches

generation_methods = ['authentic', 'dalle-3-images', 'diffusiondb', 'midjourney-images', 'midjourney_tti', 'realisticSDXL']

def predict(model, top_patches, generation_methods=generation_methods):
  model.eval()
  with torch.no_grad():
      preds = model(top_patches)

  df = pd.DataFrame(preds.numpy().round(4), columns=generation_methods)
  
  # Determine the final prediction based on majority voting
  class_counts = {method: 0 for method in generation_methods}
  for i in range(len(df)):
      predicted_class = df.iloc[i].idxmax()
      class_counts[predicted_class] += 1

  if class_counts['authentic'] > sum([class_counts[method] for method in generation_methods[1:]]):
    final_pred = 0
  else: final_pred = 1

  return final_pred


def print_report_SuSy(test_data, SuSy, class_names=None):
    
    susy_pred = []
    
    for idx in range(len(test_data)):
      top_patches = compute_top_n_patches(test_data[idx]['image'])
      susy_pred.append(predict(SuSy, top_patches))

    susy_test_labels = [0 if lbl == 0 else 1 for lbl in test_data['label']]

    cm = confusion_matrix(susy_test_labels, susy_pred)
    print(cm)

    print(classification_report(susy_test_labels, susy_pred))

    if class_names is None:
      class_names = [str(i) for i in np.unique(susy_test_labels)] 
        
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return susy_pred



# This function help us to predict the label a single image

def ext_SuSy_pred(image, SuSy):

  top_patches = compute_top_n_patches(image)
  
  return predict(SuSy, top_patches)















