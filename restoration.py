import cv2 
import numpy as np
from denoising import manualMedianFilter, manualGaussianFilter, plot_histogram
from equalization import apply_clahe_color
from sharpening import unsharpMasking

def main():
    imagePath = "input/test_image_lena_noisy.png"
    outputDir = "output"

    noisyImg = cv2.imread(imagePath)
    if noisyImg is None:
        print("gagal memuat gambar")
        return
    
    plot_histogram(noisyImg, "Input Noisy Image Histogram", f"{outputDir}/00_histogram_input.png")
    
    denoisedmedian15 = manualMedianFilter(noisyImg, 15)
    denoisedGaussian10 = manualGaussianFilter(noisyImg, 25, 10.0)
    denoised = cv2.addWeighted(denoisedmedian15, 0.5, denoisedGaussian10, 0.5, 0)
    cv2.imwrite(f"{outputDir}/01_denoised.png", denoised)
    plot_histogram(denoised, "Denoised Image Histogram", f"{outputDir}/01_histogram_denoised.png")
    

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l_clahe = apply_clahe_color(l, grid_size=(8, 8))
    a_clahe = apply_clahe_color(a, grid_size=(8, 8))
    b_clahe = apply_clahe_color(b, grid_size=(8, 8))
    
    result_lab = cv2.merge([l_clahe, a_clahe, b_clahe])
    equalized = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(f"{outputDir}/02_equalized.png", equalized)
    plot_histogram(equalized, "Equalized Image Histogram", f"{outputDir}/02_histogram_equalized.png")

    
    sharpened = unsharpMasking(equalized, 25, 10.0, 2.5)
    cv2.imwrite(f"{outputDir}/03_sharpened.png", sharpened)
    plot_histogram(sharpened, "Sharpened Image Histogram", f"{outputDir}/03_histogram_sharpened.png")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
