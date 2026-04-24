import cv2
import numpy as np

def histogramEqualization(image):

    m, n = image.shape     #ukuran gambar
    l = 256                 #range intensitas (0-255)

    hist = np.bincount(image.ravel(), minlength=l)

    pmf = hist / (m*n)

    cdf = np.cumsum(pmf)

    lut = np.round(cdf * (l - 1)).astype(np.uint8)

    imOutput = lut[image]

    return imOutput, lut


def main():
    imagePath = "output/01_denoised_combined_15-6.0.png"
    outputDir = "output"
    
    noisyImg = cv2.imread(imagePath)
    if noisyImg is None:
        print("gagal memuat gambar")
        return
    
    # histogram equalization untuk abu abu
    gray = cv2.cvtColor(noisyImg, cv2.COLOR_BGR2GRAY)
    imEq, t = histogramEqualization(gray)
    
    # histogram equalization per channel untuk color 
    b, g, r = cv2.split(noisyImg)
    b_eq, _ = histogramEqualization(b)
    g_eq, _ = histogramEqualization(g)
    r_eq, _ = histogramEqualization(r)
    imEqColor = cv2.merge((b_eq, g_eq, r_eq))
    
    # tampilkan hasil
    cv2.imshow('Input', noisyImg)
    cv2.imshow('Grayscale Equalized', imEq)
    cv2.imshow('Color Equalized', imEqColor)
    
    # simpan hasil
    cv2.imwrite(f'{outputDir}/02_equalized_gray_denoise.png', imEq)
    cv2.imwrite(f'{outputDir}/02_equalized_color_denoise.png', imEqColor)
    print("berhasil menyimpan hasil")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()