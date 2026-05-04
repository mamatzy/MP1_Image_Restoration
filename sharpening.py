import cv2
import numpy as np

from denoising import manualGaussianFilter, plot_histogram


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
            result[:, :, c] = manualLaplacianEdge(image[:, :, c])
        return result
    
    h, w = image.shape[0], image.shape[1]
    pad = kernelSize // 2
    
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
    
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernelSize, j:j+kernelSize].astype(np.float32)
            output[i, j] = np.sum(window * laplacianKernel)
    
    return output

def edgeDetectionSharpening(image, strength=1.0):
    edges = manualLaplacianEdge(image)
    
    # kombinasi: original + k * edge_mask
    imgFloat = image.astype(np.float32)
    sharpened = imgFloat + (strength * edges) 
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharpMasking(image, kernelSize, sigmaSigmaBoy, strength=1.5):
    # ngetes blur langsung pake cv2 biar cepet
    blurred = manualGaussianFilter(image, kernelSize=kernelSize, sigmaSigmaBoy=sigmaSigmaBoy)
    
    imgFloat = image.astype(np.float32)
    blurredFloat = blurred.astype(np.float32)
    
    mask = imgFloat - blurredFloat
    
    # hasil = Asli + k * Mask
    sharpened = imgFloat + (strength * mask)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def main():
    imagePath = "output/02_equalized_color_denoise_clahejadijadianpakeLAB_bilinear_15-10.0.png"
    imageTrue = "input/test_image_lena_noisy.png"
    outputDir = "output"
    
    denoisedImg = cv2.imread(imagePath)
    trueImg = cv2.imread(imageTrue)
    if denoisedImg is None or trueImg is None:
        print("gagal memuat gambar")
        return
    
    plot_histogram(denoisedImg, "Input Before Sharpening Histogram", f"{outputDir}/03_histogram_before_sharpening.png")
    
    sharpenedEdge = edgeDetectionSharpening(denoisedImg, strength=1.0)
    print("edge detection sharpening kelar")
    plot_histogram(sharpenedEdge, "Sharpened Edge Detection Histogram", f"{outputDir}/03_histogram_sharpened_edge.png")
    cv2.imwrite(f'{outputDir}/03_sharpened_laplacian.png', sharpenedEdge)
    
    sharpenedUnsharp = unsharpMasking(denoisedImg, kernelSize=25, sigmaSigmaBoy=10.0, strength=2.5)
    print("unsharp masking kelar")
    plot_histogram(sharpenedUnsharp, "Sharpened Unsharp Masking Histogram", f"{outputDir}/03_histogram_sharpened_unsharp.png")

    cv2.imshow('Edge Detection', sharpenedEdge)
    cv2.imshow('Unsharp Masking', sharpenedUnsharp)
    
    cv2.imwrite(f'{outputDir}/03_sharpened_unsharp.png', sharpenedUnsharp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
