from PIL import Image
import numpy as np
import cv2 as cv

def first():
    test_1 = cv.imread("test1.jpg" , cv.IMREAD_GRAYSCALE)

    MN = test_1.shape[0] * test_1.shape[1]
    L = 256

    histEqualization = [0] * L #All L intensities' probabilities are computed (in order to compute cdf)
    for i in range(test_1.shape[0]):
        for j in range(test_1.shape[1]):
            histEqualization[test_1[i][j]] = histEqualization[test_1[i][j]] + 1

    for i in range(L): #Computing probabilities for each gray level
        histEqualization[i] = histEqualization[i] / MN

    #Computing cdf for each intensity
    for i in range(len(histEqualization)):
        if (i != 0):
            histEqualization[i] += histEqualization[i - 1]
        else:
            histEqualization[i] = histEqualization[i]

    #Applying (L - 1)*cdf formula
    for i in range(len(histEqualization)):
        histEqualization[i] *= (L - 1)

    for i in range(len(histEqualization)):
        histEqualization[i] = np.round(histEqualization[i])

    newVals = np.zeros(shape = (L , L) , dtype=np.uint8)
    for i in range(newVals.shape[0]):
        for j in range(newVals.shape[1]):
            val = histEqualization[test_1[i][j]]
            if (val >= 1):
                newVals[i][j] = histEqualization[test_1[i][j]]

    imEqualized = cv.equalizeHist(test_1)
    
    totalDif = 0
    absDif = np.zeros(shape = (L , L) , dtype=np.uint8)
    for i in range(newVals.shape[0]): #Computing absolute difference
        for j in range(newVals.shape[1]):
            absDif[i][j] = abs(newVals[i][j] - imEqualized[i][j])
            totalDif += absDif[i][j]

    cv.imshow("output_1" , newVals)
    cv.imshow("output_2" , imEqualized)
    cv.imshow("abs(output_1 - output_2)" , absDif)
    cv.waitKey(0)

    print("Total Absolute Difference (output_1 - output_2): " , totalDif)


def second():
    test_1 = cv.imread("test1.jpg" , 0)
    test_1_arr = np.asarray(test_1)
    equalizedHist = cv.equalizeHist(test_1)

    G = 256
    MN = test_1_arr.shape[0] * test_1_arr.shape[1]
    H = [0] * G
    
    for i in range(G):
        for j in range(G):
            H[test_1_arr[i][j]] += 1

    g_min = 0
    for g in range(G):
        if (H[g] > 0):
            if (g_min > H[g]):
                g_min = H[g]

    cumulativeH = [0] * G
    cumulativeH[0] = H[0]
    for g in range(1 , G):
        cumulativeH[g] = cumulativeH[g - 1] + H[g]
    h_min = cumulativeH[g_min]

    T = [0] * G
    for g in range(G):
        T[g] = np.round((cumulativeH[g] - h_min) * (G - 1) / (MN - h_min))

    out_3 = np.zeros(shape=(test_1_arr.shape[0] , test_1_arr.shape[1]) , dtype=np.uint8)
    for i in range(test_1_arr.shape[0]):
        for j in range(test_1_arr.shape[1]):
            out_3[i][j] = T[test_1_arr[i][j]]

    cv.imshow("output_3" , out_3)

    absDif = np.abs(equalizedHist - out_3)
    cv.imshow("abs(output_2 - output_3)" , absDif)

    totalAbsDif = np.sum(absDif)
    print("Total Absolute Difference (output_3 - output_2): ", totalAbsDif)

    cv.waitKey(0)
    cv.destroyAllWindows()

first()
second()