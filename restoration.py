import cv2 
import numpy as np

img = cv2.imread("input/test_image_lena_noisy.png")

def GaussianKernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    
    x, y = np.meshgrid(ax, ax)
    
    # rumus gaussian formula
    kernel = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sigma))
    
    return kernel / np.sum(kernel)

def ManualConvolution(image, kernel):
    h, w,  = image.shape[:2]
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    output = np.zeros_like(image, dtype=np.float32)

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    for i in range(h):
        for j in range(w):
            window = padded[i : i + kernel_h, j : j + kernel_w]
            
            output[i, j] = np.sum(window * kernel)
            
    return np.clip(output, 0, 255).astype(np.uint8)


# print (GaussianKernel(5, 1.0))

while True:

    kernel_tulis_sendiri = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16

    ## abu abu

    # img_gaussian = ManualConvolution(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), GaussianKernel(51, 8.0))
    # img_gaussian2 = ManualConvolution(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), kernel_tulis_sendiri)


    # cv2.imshow("Original Image", img)
    # cv2.imshow("Gaussian Noise Image", img_gaussian)
    # cv2.imshow("Gaussian Noise Image 2", img_gaussian2)


    ## warna

    b, g, r = cv2.split(img)

    b_blur = ManualConvolution(b, GaussianKernel(51, 8.0))
    g_blur = ManualConvolution(g, GaussianKernel(51, 8.0))
    r_blur = ManualConvolution(r, GaussianKernel(51, 8.0))

    img_gaussian_color = cv2.merge((b_blur, g_blur, r_blur))

    cv2.imshow("Original Image", img)
    cv2.imshow("Gaussian Noise Image", img_gaussian_color)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()