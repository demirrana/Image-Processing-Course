import numpy as np
import cv2 as cv

print("Default absolute path of noisy image for question 1: noisyImage_Gaussian.jpg")
print("Default absolute path of noisy image for question 2: noisyImage_SaltPepper.jpg")
print("Default absolute path of clean image for both questions: lena_grayscale_hq.jpg")
print("You will be asked to enter the absolute path for the images. If you want to proceed with default, please enter 'd'.")
default = input()

def weighted_median_filter(path , filter_size):
    img_gray = cv.imread(path , cv.IMREAD_GRAYSCALE)
    M = img_gray.shape[0]
    N = img_gray.shape[1]

    pad_size = (int)((filter_size - 1) / 2)
    center_weight = filter_size

    my_filter = cv.copyMakeBorder(img_gray , pad_size , pad_size , pad_size , pad_size , borderType=cv.BORDER_REPLICATE)
    output = np.zeros(shape=(M,N) , dtype=np.uint8)

    for i in range(pad_size , M + pad_size):
        for j in range(pad_size , N + pad_size):
            if (i < pad_size or i >= M+pad_size or j < pad_size or j >= N+pad_size):
                continue
            
            arr = []
            for a in range(i - pad_size , i + pad_size + 1):
                for b in range(j - pad_size , j + pad_size + 1):
                    arr.append(my_filter[a][b])
            for a in range(center_weight - 1): #adding center pixel in the array (filter size) times.
                arr.append(my_filter[i][j])

            median_val = np.median(arr)
            output[i-pad_size][j-pad_size] = int(median_val)
    
    return output

def first():
    if (default == "d"):
        noisy_img_name = "noisyImage_Gaussian.jpg"
        clean_img_name = "lena_grayscale_hq.jpg"
    else:
        print("Please enter the absolute path of the noisy image you want to proceed with: ")
        noisy_img_name = input()
        print("\nPlease enter the absolute path of the clean image you want to proceed with: ")
        clean_img_name = input()
    
    noisy_img = cv.imread(noisy_img_name , cv.IMREAD_GRAYSCALE)
    M = noisy_img.shape[0]
    N = noisy_img.shape[1]

    pad_size = 2

    variance = 0.004

    padded_noisy_img = cv.copyMakeBorder(noisy_img , 2 , 2 , 2 , 2 , borderType=cv.BORDER_REPLICATE)
    padded_noisy_img = padded_noisy_img.astype(np.uint8)
    normalized_img = cv.normalize(padded_noisy_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    output_1_1 = np.zeros(shape=(M,N))

    for i in range(M):
        for j in range(N):
            filter_kernel = []
            for a in range(i , i + 2*pad_size + 1):      #i and j are indexed according to output image, so these for loops 
                for b in range(j , j + 2*pad_size + 1):  #are indexed according to padded image
                    filter_kernel.append(normalized_img[a][b])

            local_avr = np.mean(filter_kernel)
            local_var = np.var(filter_kernel)

            if (local_var == 0): #In the case of dividing by 0
                continue
            else:
                output_1_1[i][j] = normalized_img[i+2][j+2] - (variance / local_var) * (normalized_img[i+2][j+2] - local_avr)
            
    output_1_1  = cv.normalize(output_1_1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    output_1_1 = output_1_1.astype(np.uint8)

    output_1_2 = cv.boxFilter(noisy_img , -1 , (5 , 5) , borderType=cv.BORDER_CONSTANT)
    output_1_3 = cv.GaussianBlur(noisy_img , (5 , 5) , 0)

    clean_img = cv.imread(clean_img_name , cv.IMREAD_GRAYSCALE)

    print("PSNR value for Adaptive Mean Filter: " , cv.PSNR(clean_img , output_1_1))
    print("PSNR value for Box Filter: " , cv.PSNR(clean_img , output_1_2))
    print("PSNR value for Gaussian Filter: " , cv.PSNR(clean_img , output_1_3))

    cv.imshow("output_1_1" , output_1_1)
    cv.imshow("output_1_2" , output_1_2)
    cv.imshow("output_1_3" , output_1_3)
    cv.waitKey(0)
    cv.destroyAllWindows()

def second():
    if (default == "d"):
        noisy_img_name = "noisyImage_SaltPepper.jpg"
        clean_img_name = "lena_grayscale_hq.jpg"
    else:
        print("Please enter the absolute path of the noisy image you want to proceed with: ")
        noisy_img_name = input()
        print("\nPlease enter the absolute path of the clean image you want to proceed with: ")
        clean_img_name = input()
    
    noisy_img = cv.imread(noisy_img_name , cv.IMREAD_GRAYSCALE)
    M = noisy_img.shape[0]
    N = noisy_img.shape[1]

    #I padded with 3 sequences in each way since we may use 7x7 kernel
    padded_noisy_img = cv.copyMakeBorder(noisy_img , 3 , 3 , 3 , 3 , borderType=cv.BORDER_REPLICATE)
    padded_noisy_img = padded_noisy_img.astype(np.uint8)
    normalized_img = cv.normalize(padded_noisy_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    output_2_1 = np.zeros(shape=(M,N))

    for i in range(M):
        for j in range(N):
            filter_kernel = []
            S_xy = 3
            S_max = 7
            pad_size = 1

            while (S_xy <= 7): #for each kernel size, it computes the kernels and looks at the conditions
                for a in range(i , i + 2*pad_size + 1):      #i and j are indexed according to output image, so these for loops 
                    for b in range(j , j + 2*pad_size + 1):  #are indexed according to padded image
                        filter_kernel.append(normalized_img[a][b])
        
                z_min = np.min(filter_kernel)
                z_med = np.median(filter_kernel)
                z_max = np.max(filter_kernel)
                z_xy = normalized_img[i + pad_size][j + pad_size] #center pixel of the kernel

                if (z_min < z_med < z_max): #level A first condition
                    if (z_min < z_xy < z_max): #level B
                        output_2_1[i][j] = z_xy
                    else:  #level B
                        output_2_1[i][j] = z_med
                    break
                elif (S_xy < 7): #level A second and third condition
                    S_xy += 2  #Since this is a while loop, kernel size increase will work
                else: #level A last condition
                    output_2_1[i][j] = z_med
                    break
            
    output_2_1  = cv.normalize(output_2_1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    output_2_1 = output_2_1.astype(np.uint8)

    output_2_2 = cv.medianBlur(noisy_img , 3)
    output_2_3 = cv.medianBlur(noisy_img , 5)
    output_2_4 = cv.medianBlur(noisy_img , 7)
    output_2_5 = weighted_median_filter(noisy_img_name , 3)
    output_2_6 = weighted_median_filter(noisy_img_name , 5)
    output_2_7 = weighted_median_filter(noisy_img_name , 7)

    clean_img = cv.imread(clean_img_name , cv.IMREAD_GRAYSCALE)

    print("PSNR value for Adaptive Median Filter: " , cv.PSNR(clean_img , output_2_1))
    print("PSNR value for Median Filter(3x3): " , cv.PSNR(clean_img , output_2_2))
    print("PSNR value for Median Filter(5x5): " , cv.PSNR(clean_img , output_2_3))
    print("PSNR value for Median Filter(7x7): " , cv.PSNR(clean_img , output_2_4))
    print("PSNR value for Center Weighted Median Filter(3x3): " , cv.PSNR(clean_img , output_2_5))
    print("PSNR value for Center Weighted Median Filter(5x5): " , cv.PSNR(clean_img , output_2_6))
    print("PSNR value for Center Weighted Median Filter(7x7): " , cv.PSNR(clean_img , output_2_7))

    cv.imshow("output_2_1" , output_2_1)
    cv.imshow("output_2_2" , output_2_2)
    cv.imshow("output_2_3" , output_2_3)
    cv.imshow("output_2_4" , output_2_4)
    cv.imshow("output_2_5" , output_2_5)
    cv.imshow("output_2_6" , output_2_6)
    cv.imshow("output_2_7" , output_2_7)
    cv.waitKey(0)
    cv.destroyAllWindows()

print("\nQUESTION 1\n")
first()
print("\nQUESTION 2\n")
second()