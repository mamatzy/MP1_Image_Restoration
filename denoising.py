import cv2
import numpy as np


def manualMedianFilter(image, kernelSize):
    # ============== Referensi kode =========== 
    # ==== Kode aseli dari https://medium.com/@sarves021999/noise-filtering-mean-median-mid-point-filter-72ab3be76da2 ====
    # def medianFilter(orginalImg, wrappedImage, kernelSize : int):
    #     filteredImage = np.zeros(orginalImg.shape,dtype=np.int32)
    #     image_h, image_w = orginalImg.shape[0], orginalImg.shape[1]
    #     w = kernelSize//2

    #     for i in range(w, image_h - w): ## traverse image row
    #         for j in range(w, image_w - w):  ## traverse image col 
    #             overlapImg = wrappedImage[i-w : i+w+1, j-w : j+w+1 ]    # Crop image for mask product         
    #             filteredImage[i][j] = np.median(overlapImg.reshape(-1, 3), axis=0)  # Filtering
                
    #     return filteredImage

    # beda dikit karena buat yang berwarna
    if len(image.shape) == 3:  
        result = np.zeros_like(image, dtype=np.uint8)
        for c in range(image.shape[2]):
            result[:, :, c] = manualMedianFilter(image[:, :, c], kernelSize)
        return result
    
    h, w = image.shape[0], image.shape[1]
    pad = kernelSize // 2
    # beri padding refleksi buat pixel paling pinggir
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
    
    output = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernelSize, j:j+kernelSize]
            output[i, j] = np.median(window)
    
    return output


def gaussianKernel(size, sigmaSigmaBoy):

    #============== Referensi kode ===========
    # ==== Kode aseli dari https://www.kaggle.com/code/dasmehdixtr/gaussian-filter-implementation-from-scratch ====
    # def gkernel(l=3, sig=2):
    #     """\
    #     Gaussian Kernel Creator via given length and sigmaSigmaBoy
    #     """

    #     ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    #     xx, yy = np.meshgrid(ax, ax)

    #     kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    #     return kernel / np.sum(kernel)

    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x, y = np.meshgrid(ax, ax)
    
    kernel = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sigmaSigmaBoy))
    
    return kernel / np.sum(kernel)


def manualGaussianFilter(image, kernelSize, sigmaSigmaBoy):
    # ngikut kek yang di manualMedianFilter karena buat yang berwarna, sama konvolusi pake gaussian kernel 
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.uint8)
        for c in range(image.shape[2]):
            result[:, :, c] = manualGaussianFilter(image[:, :, c], kernelSize, sigmaSigmaBoy)
        return result
    
    h, w = image.shape[0], image.shape[1]
    pad = kernelSize // 2
    
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
    
    kernel = gaussianKernel(kernelSize, sigmaSigmaBoy)
    
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernelSize, j:j+kernelSize].astype(np.float32)
            output[i, j] = np.sum(window * kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)


def main():
    imagePath = "input/test_image_lena_noisy.png"
    outputDir = "output"
    
    noisyImg = cv2.imread(imagePath)
    if noisyImg is None:
        print("gagal ngambi gambar")
        return
    
    
    # median filter
    # denoisedMedian10 = manualMedianFilter(noisyImg, kernelSize=10)
    denoisedMedian15 = manualMedianFilter(noisyImg, kernelSize=15)
    # denoisedMedian25 = manualMedianFilter(noisyImg, kernelSize=25)
    print("median filter kelar")
    
    # gaussian filter
    # denoisedGaussian4 = manualGaussianFilter(noisyImg, kernelSize=25, sigmaSigmaBoy=4.0)
    denoisedGaussian6 = manualGaussianFilter(noisyImg, kernelSize=25, sigmaSigmaBoy=6.0)
    # denoisedGaussian10 = manualGaussianFilter(noisyImg, kernelSize=25, sigmaSigmaBoy=10.0)
    # denoisedGaussian20 = manualGaussianFilter(noisyImg, kernelSize=25, sigmaSigmaBoy=20.0)
    print("gaussian filter kelar")
    
    # kombinasi hasil
    denoised = cv2.addWeighted(denoisedMedian15, 0.5, denoisedGaussian6, 0.5, 0)
    print("yahahaha kelar semua")
    
    # i show speed typeshii
    # cv2.imshow('Median Filter 10', denoisedMedian10)
    cv2.imshow('Median Filter 15', denoisedMedian15)
    # cv2.imshow('Median Filter 25', denoisedMedian25)
    # cv2.imshow('Gaussian Filter 4.0', denoisedGaussian4)
    cv2.imshow('Gaussian Filter 6.0', denoisedGaussian6)
    # cv2.imshow('Gaussian Filter 20.0', denoisedGaussian20)
    cv2.imshow('Combined', denoised)
    
    # simpan hasil
    # cv2.imwrite(f'{outputDir}/01_denoised_median_10.png', denoisedMedian10)
    cv2.imwrite(f'{outputDir}/01_denoised_median_15.png', denoisedMedian15)
    # cv2.imwrite(f'{outputDir}/01_denoised_median_25.png', denoisedMedian25)
    # cv2.imwrite(f'{outputDir}/01_denoised_gaussian_4.0.png', denoisedGaussian4)
    cv2.imwrite(f'{outputDir}/01_denoised_gaussian_6.0.png', denoisedGaussian6)
    # cv2.imwrite(f'{outputDir}/01_denoised_gaussian_20.0.png', denoisedGaussian20)
    cv2.imwrite(f'{outputDir}/01_denoised_combined_15-6.0.png', denoised)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
