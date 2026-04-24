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

def apply_clahe_color(img):
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # tanyain ntar boleh pake createCLAHE atau engga ke pak Arta
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a_channel, b_channel))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_img


def main():
    versi = "15-10.0" # paling bagus 15-10.0
    imagePath = f"output/01_denoised_combined_{versi}.png"
    outputDir = "output"
    
    noisyImg = cv2.imread(imagePath)
    if noisyImg is None:
        print("gagal memuat gambar")
        return
    
    # histogram equalization & clahe untuk abu abu
    gray = cv2.cvtColor(noisyImg, cv2.COLOR_BGR2GRAY)
    imEq, _ = histogramEqualization(gray)

    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe_obj.apply(gray)
    
    # histogram equalization & clahe per channel untuk color 
    b, g, r = cv2.split(noisyImg)
    b_eq, _ = histogramEqualization(b)
    g_eq, _ = histogramEqualization(g)
    r_eq, _ = histogramEqualization(r)
    imEqColor = cv2.merge((b_eq, g_eq, r_eq))

    imEqColor_CLAHE = apply_clahe_color(noisyImg)
    
    # # tampilkan hasil
    # cv2.imshow('Input', noisyImg)
    # cv2.imshow('Grayscale Equalized', imEq)
    # cv2.imshow('Grayscale Equalized with CLAHE', clahe_gray)
    cv2.imshow('Color Equalized', imEqColor)
    cv2.imshow('Color Equalized with CLAHE', imEqColor_CLAHE)
    # # simpan hasil
    # cv2.imwrite(f'{outputDir}/02_equalized_gray_denoise.png', imEq)
    # cv2.imwrite(f'{outputDir}/02_equalized_color_denoise.png', imEqColor)
    cv2.imwrite(f'{outputDir}/02_equalized_color_denoise_clahe_{versi}.png', imEqColor_CLAHE)
    print("berhasil menyimpan hasil")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()