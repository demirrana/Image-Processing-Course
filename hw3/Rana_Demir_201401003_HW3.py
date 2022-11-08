import cv2 as cv
import numpy as np

print("Default clean image path: lena_grayscale_hq.jpg")
print("Default noisy image path: noisyImage.jpg")
print("In the following lines, you will be asked to enter the absolute path of the clean and noisy images. If you want to proceed with the default as given in the questions, please enter 'd':")
default = input()

#Finds the median filtered array and returns it and openCV version. Also if weighted parameter is given as True, it finds the center weighted median filter.
def medianFilter(path , weighted=False):
    img_gray = cv.imread(path , cv.IMREAD_GRAYSCALE)
    M = img_gray.shape[0]
    N = img_gray.shape[1]

    filter_size = 5

    median_filter = cv.medianBlur(img_gray , filter_size)

    my_filter = cv.copyMakeBorder(img_gray , 2 , 2 , 2 , 2 , borderType=cv.BORDER_REPLICATE)
    output = np.zeros(shape=(M,N) , dtype=np.uint8)

    for i in range(2 , M + 2):
        for j in range(2 , N + 2):
            if (i < 2 or i >= M+2 or j < 2 or j >= N+2):
                continue

            arr = [my_filter[i-2][j-2] , my_filter[i-2][j-1] , my_filter[i-2][j] , my_filter[i-2][j+1] , my_filter[i-2][j+2] ,
                my_filter[i-1][j-2] , my_filter[i-1][j-1] , my_filter[i-1][j] , my_filter[i-1][j+1] , my_filter[i-1][j+2] ,
                my_filter[i][j-2] , my_filter[i][j-1] , my_filter[i][j] , my_filter[i][j+1] , my_filter[i][j+2] ,
                my_filter[i+1][j-2] , my_filter[i+1][j-1] , my_filter[i+1][j] , my_filter[i+1][j+1] , my_filter[i+1][j+2] ,
                my_filter[i+2][j-2] , my_filter[i+2][j-1] , my_filter[i+2][j] , my_filter[i+2][j+1] , my_filter[i+2][j+2]]
            
            if (weighted == True):
                arr.append(my_filter[i][j])
                arr.append(my_filter[i][j])

            median_val = np.median(arr)
            output[i-2][j-2] = int(median_val)

    return output , median_filter

def first():
    if (default == "d"):
        path = "noisyImage.jpg"
    else:
        print("Please enter the absolute path of the image that you want to proceed with:")
        path = input()

    output , median_filter = medianFilter(path)

    abs_difference = np.abs(output - median_filter)
    print("The summation of the absolute difference matrix:" , np.sum(abs_difference))

    cv.imshow("my median filter" , output)
    cv.imshow("opencv median filter" , median_filter)
    cv.waitKey(0)
    cv.destroyAllWindows()

def second():
    if (default == "d"):
        clean_img_name = "lena_grayscale_hq.jpg"
        noisy_img_name = "noisyImage.jpg"
    else:
        print("Please enter the absolute path of the image that you will use as clean version:")
        clean_img_name = input()
        print("Please enter the absolute path of the image that you will use as noisy version:")
        noisy_img_name = input()
    
    clean_img = cv.imread(clean_img_name , cv.IMREAD_GRAYSCALE)
    noisy_img = cv.imread(noisy_img_name , cv.IMREAD_GRAYSCALE)

    box_filtered = cv.boxFilter(noisy_img , -1 , (5 , 5) , borderType=cv.BORDER_CONSTANT)
    gaussian_filtered = cv.GaussianBlur(noisy_img , (7 , 7) , 0)
    median_filtered = cv.medianBlur(noisy_img , 5)

    print("The PSNR for box filter: " , cv.PSNR(clean_img , box_filtered))
    print("The PSNR for gaussian filter: " , cv.PSNR(clean_img , gaussian_filtered))
    print("The PSNR for median filter: " , cv.PSNR(clean_img , median_filtered))


def third():
    if (default == "d"):
        clean_img_name = "lena_grayscale_hq.jpg"
        noisy_img_name = "noisyImage.jpg"
    else:
        print("Please enter the absolute path of the image that you will use as clean version:")
        clean_img_name = input()
        print("Please enter the absolute path of the image that you will use as noisy version:")
        noisy_img_name = input()

    clean_img = cv.imread(clean_img_name , cv.IMREAD_GRAYSCALE)
    noisy_img = cv.imread(noisy_img_name , cv.IMREAD_GRAYSCALE)
    my_median_filter , median_filter = medianFilter(noisy_img_name)
    weighted_median_filter , median_filter = medianFilter(noisy_img_name , True)
    box_filtered = cv.boxFilter(noisy_img , -1 , (5 , 5) , borderType=cv.BORDER_CONSTANT)
    gaussian_filtered = cv.GaussianBlur(noisy_img , (7 , 7) , 0)

    print("PSNR for my own median filter:" , cv.PSNR(clean_img , my_median_filter))
    print("PSNR for OpenCV's Box filter:" , cv.PSNR(clean_img , box_filtered))
    print("PSNR for OpenCV's Gaussian filter:" , cv.PSNR(clean_img , gaussian_filtered))
    print("PSNR for OpenCV's Median filter:" , cv.PSNR(clean_img , median_filter))
    print("PSNR for my center weighted median filter:" , cv.PSNR(clean_img , weighted_median_filter))

    cv.imshow("Own Median Filter" , my_median_filter)
    cv.imshow("OpenCV Box Filter" , box_filtered)
    cv.imshow("OpenCV Gaussian Filter" , gaussian_filtered)
    cv.imshow("OpenCV Median Filter" , median_filter)
    cv.imshow("Own Center Weighted Median Filter" , weighted_median_filter)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return weighted_median_filter


def fourth(center_weighted_median_filter):
    if (default == "d"):
        clean_img_name = "lena_grayscale_hq.jpg"
    else:
        print("Please enter the absolute path of the image that you will use as clean version:")
        clean_img_name = input()

    clean_img = cv.imread(clean_img_name , cv.IMREAD_GRAYSCALE)

    new_arr = cv.transpose(center_weighted_median_filter)

    print("PSNR for my new image:" , cv.PSNR(clean_img , new_arr))
    print("PSNR for my center weighted median filter:" , cv.PSNR(clean_img , center_weighted_median_filter))

    cv.imshow("clean image" , clean_img)
    cv.imshow("center weighted median filter" , center_weighted_median_filter)     
    cv.imshow("new adjusted image (transpose)" , new_arr)
    cv.waitKey(0)
    cv.destroyAllWindows()


print("First Question:\n")
first()
print("\nSecond Question:\n")
second()
print("\nThird Question:\n")
center_weighted_median_filter = third() #center weighted median filter array is returned in the function "third"
print("\nFourth Question:\n")
fourth(center_weighted_median_filter)