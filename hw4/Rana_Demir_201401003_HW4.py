from code import interact
import cv2 as cv
import numpy as np

print("Default image path: lena_grayscale_hq.jpg")
print("In the following lines, you will be asked to enter the absolute path of the image. If you want to proceed with the default as given in the questions, please enter 'd':")
default = input()

def first():
    if (default == "d"):
        img_path = "lena_grayscale_hq.jpg"
    else:
        print("Please enter the absolute path of the image you want to proceed with for the 1st question:")
        img_path = input()

    img = cv.imread(img_path , cv.IMREAD_GRAYSCALE)
    integral_img = cv.integral(img)
    M = img.shape[0]
    N = img.shape[1]

    padded_img = np.zeros(shape=(M+1,N+1))
    #Making a padded image from top and left sides of the image we use as source
    for x in range(M):
        for y in range(N):
            padded_img[x+1][y+1] = img[x][y]

    #Integral image will be computed from the copy of the padded image
    my_integral_img = padded_img.copy()

    for x in range(M):
        for y in range(N):
            my_integral_img[x+1][y+1] = my_integral_img[x][y+1] + my_integral_img[x+1][y] - my_integral_img[x][y] + padded_img[x+1][y+1]
    #Difference is computed
    different_pixels = 0
    difference = integral_img - my_integral_img
    for a in range(M):
        for b in range(N):
            difference[a][b] = 100 * (integral_img[a][b] - my_integral_img[a][b])
            if (difference[a][b] != 0):
                different_pixels = different_pixels + 1

    print("The number of different pixels: " , different_pixels)
    print("\nDifference array:\n" , difference)

def second():
    filter_size = 3
    filter_pixels = 9
    
    if (default == "d"):
        img_path = "lena_grayscale_hq.jpg"
    else:
        print("Please enter the absolute path of the image you want to proceed with for the 2nd question:")
        img_path = input()

    img = cv.imread(img_path , cv.IMREAD_GRAYSCALE)
    integral_img = cv.integral(img)
    M = img.shape[0]
    N = img.shape[1]

    #I added zero padding to integral image, so that it can have 2 sequences of zeros for all top, bottom, left and right
    #(Because it will be needed to compute the integral image) 
    padded_integral_img = cv.copyMakeBorder(integral_img , 1 , 2 , 1 , 2 , borderType=cv.BORDER_CONSTANT)
    copy_integral = padded_integral_img.copy()
    my_filtered_img = padded_integral_img.copy()

    #Four integers that are added: bottom right + top left - top right - bottom left (the last 3 are from the outside of the filter)
    #Then, this sum is divided to total num of pixels which is 9 for this case (one floating point division)
    for i in range(2,M+2):
        for j in range(2,N+2):
            if (i == M+1 and j == N+1):
                four_int_sum = copy_integral[i][j] + copy_integral[i-2][j-2] - copy_integral[i-2][j] - copy_integral[i][j-2]
            elif (i == M+1):
                four_int_sum = copy_integral[i][j+1] + copy_integral[i-2][j-2] - copy_integral[i-2][j+1] - copy_integral[i][j-2]
            elif (j == N+1):
                four_int_sum = copy_integral[i+1][j] + copy_integral[i-2][j-2] - copy_integral[i-2][j] - copy_integral[i+1][j-2]
            else:
                four_int_sum = copy_integral[i+1][j+1] + copy_integral[i-2][j-2] - copy_integral[i-2][j+1] - copy_integral[i+1][j-2]
            
            one_float_point_division = four_int_sum / filter_pixels
            my_filtered_img[i][j] = np.round(one_float_point_division)

    implemented_box_filter = np.zeros(shape=(M,N) , dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            implemented_box_filter[i][j] = my_filtered_img[i+2][j+2]

    opencv_box_filter = cv.blur(img, (3,3), borderType = cv.BORDER_CONSTANT)
    padded_opencv_box_filter = cv.copyMakeBorder(opencv_box_filter , 2 , 2 , 2 , 2 , borderType=cv.BORDER_CONSTANT)

    different_pixels = 0 #num of pixels that have difference more than 3
    difference = np.zeros(shape=(M+2,N+2))
    #I took the pixels
    for i in range(2,M+2):
        for j in range(2,N+2):
            difference[i][j] = np.abs(padded_opencv_box_filter[i][j] - my_filtered_img[i][j])
            if (difference[i][j] >= 3):
                different_pixels += 1

    my_filtered_img = my_filtered_img.astype(np.uint8)

    print("\nThe number of pixels that have difference more than 3 for openCV's and integral image implementation of box filter:" , different_pixels)
    print("\nDifference array:\n" , difference)

    cv.imshow("openCV box filter" , opencv_box_filter)
    cv.imshow("box filter using integral image" , implemented_box_filter)
    cv.waitKey(0)
    cv.destroyAllWindows()

print("\nQUESTION 1\n")
first()
print("\nQUESTION 2\n")
second()