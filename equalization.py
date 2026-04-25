import cv2
import numpy as np

def histogramEqualization(image, clip_limit=0.01):

    m, n = image.shape     #ukuran gambar
    l = 256                 #range intensitas (0-255)

    hist = np.bincount(image.ravel(), minlength=l)

    pmf = hist / (m*n)

    clipping_mask = pmf > clip_limit
    excess = np.sum(pmf[clipping_mask] - clip_limit)
    pmf[clipping_mask] = clip_limit
    pmf += excess / l # Distribusikan kembali kelebihan nilai agar total tetap 1.0

    cdf = np.cumsum(pmf)

    lut = np.round(cdf * (l - 1)).astype(np.uint8)

    # ngembaliin lut tanpa langsung ke gambar soale mau ku itung lut per grid
    return lut

def apply_clahe_color(img, grid_size):
    
    h, w = img.shape
    grid_rows, grid_cols = grid_size
    
    tile_h = h // grid_rows
    tile_w = w // grid_cols
    
    result = np.zeros_like(img)
    # List untuk menyimpan LUT setiap grid
    grid_luts = []
    
    # looping untuk histogram equalization per grid
    for i in range(grid_rows):
        row_luts = []
        for j in range(grid_cols):
            # ambil koordinat grid
            y1 = i * tile_h
            y2 = h if i == grid_rows - 1 else (i + 1) * tile_h
            x1 = j * tile_w
            x2 = w if j == grid_cols - 1 else (j + 1) * tile_w
            
            tile = img[y1:y2, x1:x2]
            
            row_luts.append(histogramEqualization(tile))

        grid_luts.append(row_luts)

    # looping untuk bilinear interpolation 
    for y in range(h):
        for x in range(w):
            # posisi relatif piksel terhadap pusat grid
            gf_y = (y - tile_h / 2) / tile_h
            gf_x = (x - tile_w / 2) / tile_w

            r1 = int(np.floor(gf_y))
            r2 = r1 + 1
            c1 = int(np.floor(gf_x))
            c2 = c1 + 1

            r1, r2 = np.clip([r1, r2], 0, grid_rows - 1)
            c1, c2 = np.clip([c1, c2], 0, grid_cols - 1)

            dy = gf_y - r1
            dx = gf_x - c1

            pixel_val = img[y, x]

            val11 = grid_luts[r1][c1][pixel_val]
            val12 = grid_luts[r1][c2][pixel_val]
            val21 = grid_luts[r2][c1][pixel_val]
            val22 = grid_luts[r2][c2][pixel_val]

            # rumus Bilinear Interpolation
            interpolated_val = (
                val11 * (1 - dx) * (1 - dy) +
                val12 * dx * (1 - dy) +
                val21 * (1 - dx) * dy +
                val22 * dx * dy
            )

            result[y, x] = interpolated_val

    return result


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
    imEq = histogramEqualization(gray)

    clahe_gray = apply_clahe_color(gray, grid_size=(8, 8))
    
    # histogram equalization & clahe per channel untuk color 
    b, g, r = cv2.split(noisyImg)
    b_eq = histogramEqualization(b)
    g_eq = histogramEqualization(g)
    r_eq = histogramEqualization(r)
    imEqColor = cv2.merge((b_eq, g_eq, r_eq))

    b_cl = apply_clahe_color(b, grid_size=(8, 8))
    g_cl = apply_clahe_color(g, grid_size=(8, 8))
    r_cl = apply_clahe_color(r, grid_size=(8, 8))
    # imEqColor_CLAHE = cv2.merge((b_cl, g_cl, r_cl))

    # CLAHE pake LAB color space
    lab = cv2.cvtColor(noisyImg, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Terapkan fungsi grid Anda hanya pada channel L
    l_clahe = apply_clahe_color(l, grid_size=(8, 8)) 

    # Gabung kembali
    result_lab = cv2.merge((l_clahe, a, b))
    imEqColor_CLAHE = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    # # tampilkan hasil
    # cv2.imshow('Input', noisyImg)
    # cv2.imshow('Grayscale Equalized', imEq)
    # cv2.imshow('Grayscale Equalized with CLAHE', clahe_gray)
    cv2.imshow('Color Equalized', imEqColor)
    cv2.imshow('Color Equalized with CLAHE', imEqColor_CLAHE)
    cv2.imshow('Color Equalized with CLAHE + Bilinear', imEqColor_CLAHE)
    # # simpan hasil
    # cv2.imwrite(f'{outputDir}/02_equalized_gray_denoise.png', imEq)
    # cv2.imwrite(f'{outputDir}/02_equalized_color_denoise.png', imEqColor)
    # cv2.imwrite(f'{outputDir}/02_equalized_color_denoise_clahejadijadianpakeLAB_{versi}.png', imEqColor_CLAHE)
    cv2.imwrite(f'{outputDir}/02_equalized_color_denoise_clahejadijadianpakeLAB_bilinear_{versi}.png', imEqColor_CLAHE)

    print("berhasil menyimpan hasil")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()