import cv2 as cv
import numpy as np
from os.path import exists

def box_filter(filter_size , path):

    img = cv.imread(path)
    img_gray = cv.imread(path , cv.IMREAD_GRAYSCALE)
    img_arr = np.asarray(img_gray)  #assuming image is of size M x N
    M = img_arr.shape[0]
    N = img_arr.shape[1]

    out_img_arr = np.zeros(shape = (M , N) , dtype=np.uint8)

    filtered_img = cv.boxFilter(img_gray , -1 , (filter_size , filter_size) , borderType=cv.BORDER_CONSTANT)
    print("\n\nBOX FILTER:\n" , filtered_img)
    filtered_arr = np.asarray(filtered_img)

    pad_size = int((filter_size - 1) / 2)    #number of zeros to be added in each direction for zero padding
    padded_arr = np.zeros(shape=(M + 2 * pad_size , N + pad_size * 2) , dtype=np.uint8) #zero padding applied array

    for i in range(pad_size , M + pad_size):
        for j in range(pad_size , N + pad_size):
            padded_arr[i][j] = img_arr[i - pad_size][j - pad_size]

    filter_pixels = filter_size * filter_size   #number of total pixels in the box filter (9 for 3x3)

    i = pad_size #row
    j = pad_size #column
    while (i != (M + pad_size) or j != (N + pad_size)):
        if (j == (N + pad_size)):
            j = pad_size
            i = i + 1 #continues with the next row
            continue

        if (i == (M + pad_size)): #breaks the loop when the bound is exceeded
            break
        
        sum = 0
        for k in range(i - pad_size , i + pad_size + 1):
            for l in range(j - pad_size , j + pad_size + 1):
                sum += int(padded_arr[k][l])

        out_img_arr[i - pad_size][j - pad_size] = np.round(int(sum) / (filter_pixels))

        j = j + 1 #continues with the next column for that row

    difference_arr = np.zeros(shape=(M,N) , dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            if (np.abs(filtered_arr[i][j] - out_img_arr[i][j]) < 0): #preventing -1 occurring as 255
                difference_arr[i][j] = -1 * (filtered_arr[i][j] - out_img_arr[i][j])
            else:
                difference_arr[i][j] = np.abs(filtered_arr[i][j] - out_img_arr[i][j])

    print("Absolute difference for filter size" , filter_size , ":\n" , difference_arr)

    return out_img_arr , filtered_arr


def separable_box_filter(filter_size , path):

    img_gray = cv.imread(path , cv.IMREAD_GRAYSCALE)
    img_arr = np.asarray(img_gray)  #assuming image is of size M x N
    M = img_arr.shape[0]
    N = img_arr.shape[1]

    out_img_arr = np.zeros(shape = (M , N) , dtype=np.uint8)

    filtered_img = cv.boxFilter(img_gray , -1 , (filter_size , filter_size) , borderType=cv.BORDER_CONSTANT)
    filtered_arr = np.asarray(filtered_img)

    pad_size = int((filter_size - 1) / 2)    #number of zeros to be added
    padded_arr = np.zeros(shape=(M + 2 * pad_size , N + pad_size * 2))

    for i in range(pad_size , M + pad_size):
        for j in range(pad_size , N + pad_size):
            padded_arr[i][j] = img_arr[i - pad_size][j - pad_size]

    #Multiplying by row filter
    for i in range(pad_size , M + pad_size):
        for j in range(pad_size , N + pad_size):

            sum = 0

            for k in range(j - pad_size , j + pad_size + 1):
                sum += padded_arr[i][k]

            out_img_arr[i - pad_size][j - pad_size] = sum / filter_size

    for i in range(pad_size , M + pad_size): #updating the matrix that has zero padding
        for j in range(pad_size , N + pad_size):
            padded_arr[i][j] = out_img_arr[i - pad_size][j - pad_size]

    out_img_arr = np.zeros(shape = (M , N) , dtype=np.uint8) #making the output matrix zero again in order to make summations correctly

    #Multiplying by column filter
    for i in range(pad_size , M + pad_size):
        for j in range(pad_size , N + pad_size):

            sum = 0

            for k in range(i - pad_size , i + pad_size + 1):
                sum += padded_arr[k][j]

            out_img_arr[i - pad_size][j - pad_size] = int(np.round(sum / filter_size))

    difference_arr = np.zeros(shape=(M,N) , dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            if (np.abs(filtered_arr[i][j] - out_img_arr[i][j]) < 0): #preventing -1 occurring as 255
                difference_arr[i][j] = -1 * (filtered_arr[i][j] - out_img_arr[i][j])
            else:
                difference_arr[i][j] = np.abs(filtered_arr[i][j] - out_img_arr[i][j])

    print("\nAbsolute difference (separable) for filter size " , filter_size , ":\n" , difference_arr)

    return out_img_arr

print("Please enter the absolute path of the image that you want to proceed with:")
path = input()
while (not exists(path)):
    print("There exists no such a file. Please reenter:")
    path = input()

img = cv.imread(path)
cv.imshow("original image" , img)

print("\nPlease enter the first filter size you want to apply Box Filter: ")
filter_size_1 = int(input())
output_1_1 , output_2_1 = box_filter(filter_size_1 , path)

print("\nPlease enter the second filter size you want to apply Box Filter: ")
filter_size_2 = int(input())
output_1_2 , output_2_2 = box_filter(filter_size_2 , path)

print("\nPlease enter the third filter size you want to apply Box Filter: ")
filter_size_3 = int(input())
output_1_3 , output_2_3= box_filter(filter_size_3 , path)

output_3_1 = separable_box_filter(filter_size_1 , path)

output_3_2 = separable_box_filter(filter_size_2 , path)

output_3_3 = separable_box_filter(filter_size_3 , path)

cv.imshow("output_1_1" , output_1_1)
cv.imshow("output_1_2" , output_1_2)
cv.imshow("output_1_3" , output_1_3)
cv.imshow("output_2_1" , output_2_1)
cv.imshow("output_2_2" , output_2_2)
cv.imshow("output_2_3" , output_2_3)
cv.imshow("output_3_1" , output_3_1)
cv.imshow("output_3_2" , output_3_2)
cv.imshow("output_3_3" , output_3_3)
cv.waitKey(0)
cv.destroyAllWindows()