import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca_image_reconstruction(image_path, n_components):
    # Step 1: Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Standardize the image (zero mean and unit variance) - each pixel as a feature
    image_normalized = (image - np.mean(image)) / np.std(image)

    # Step 3: Initialize PCA with the desired number of components
    pca = PCA(n_components=n_components)

    # Step 4: Apply PCA (fit and transform the image data)
    # We treat each row of the image as a sample, and each column as a feature.
    image_pca = pca.fit_transform(image_normalized)

    # Step 5: Inverse transform the PCA result to reconstruct the image
    image_reconstructed_normalized = pca.inverse_transform(image_pca)

    # Step 6: Reverse the standardization
    image_reconstructed = image_reconstructed_normalized * np.std(image) + np.mean(image)

    # Step 7: Display the images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_reconstructed, cmap='gray')
    plt.title(f"Reconstructed Image with {n_components} Components")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.plot(image.flatten()[:100], label="Original Image Pixels")
    plt.plot(image_reconstructed.flatten()[:100], label="Reconstructed Image Pixels", linestyle='dashed')
    plt.legend()
    plt.title("Pixel Comparison (First 100 Pixels)")

    plt.show()

    return image_reconstructed


# Example usage
image_path = 'my.jpg'  # Replace with your image path
n_components = 50  # Number of principal components to retain
reconstructed_image = pca_image_reconstruction(image_path, n_components)
