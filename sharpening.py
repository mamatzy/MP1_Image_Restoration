import cv2
import numpy as np


def manualLaplacianEdge(image):

    kernelSize = 3

    laplacianKernel = np.array([
        [0, -1, 0],
        [-1, 4  , -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # ngikut kek yang di manualMedianFilter karena buat yang berwarna, sama konvolusi pake gaussian kernel 
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.uint8)
        for c in range(image.shape[2]):
            result[:, :, c] = manualLaplacianEdge(image[:, :, c], kernelSize)
        return result
    
    h, w = image.shape[0], image.shape[1]
    pad = kernelSize // 2
    
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
    
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernelSize, j:j+kernelSize].astype(np.float32)
            output[i, j] = np.sum(window * laplacianKernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)


def edgeDetectionSharpening(image, strength=1.0):
    # edge detection sharpening: hasil = asli + k * mask_edge
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.uint8)
        for c in range(image.shape[2]):
            result[:, :, c] = edgeDetectionSharpening(image[:, :, c], strength)
        return result
    
    edges = manualLaplacianEdge(image)
    
    # normalisasi edges ke range 0-1
    edgeMask = edges.astype(np.float32) / 255.0
    
    # kombinasi: original + k * edge_mask
    imgFloat = image.astype(np.float32)
    sharpened = imgFloat + strength * (edgeMask * 255)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def main():
    imagePath = "output/01_denoised_combined_15-10.0.png"
    outputDir = "output"
    
    noisyImg = cv2.imread(imagePath)
    if noisyImg is None:
        print("gagal memuat gambar")
        return
    
    sharpenedEdge = edgeDetectionSharpening(noisyImg, strength=6.0)
    print("edge detection sharpening kelar")
    
    cv2.imshow('Edge Detection', sharpenedEdge)
    
    cv2.imwrite(f'{outputDir}/03_sharpened_laplacian_denoisedlu_3-6.0.png', sharpenedEdge)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
