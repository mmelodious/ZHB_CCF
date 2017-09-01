# coding=utf-8
import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
#import glob
from sklearn.ensemble import RandomForestClassifier
SpectralPath = ["./train/sao_paulo/landsat_8/LC82190762013244LGN00"]
GTFile = ["./train/sao_paulo/lcz/sao_paulo_lcz_GT.tif"]
# for i in glob.glob(os.path.join(BerlinPath, "*.tif")):
#     BerlinFile.append(i)
# BerlinFile.sort()

def ReadCity(FilePath, GT, SelectPer):
    SpectralFile = os.listdir(FilePath)
    SpectralFile.sort()
    print SpectralFile
    GTImage = skimage.io.imread(GT)
    ImageSize = GTImage.shape
    SpectralImage = np.array([])
    SpectralImage.shape = ImageSize[0], ImageSize[1], 0
    for k in SpectralFile:
        FileName = os.path.join(FilePath, k)
        Image0 = skimage.io.imread(FileName)
        Image1 = Image0[Image0 != -99999]
        ImageMean = np.mean(Image1)
        ImageStd = np.linalg.norm(Image1-ImageMean, 2)*1.0/np.sqrt((Image1.size-1))
        Image0 = (Image0-ImageMean)*1.0/ImageStd


        Image0.shape = ImageSize[0], ImageSize[1], 1
        SpectralImage = np.concatenate((SpectralImage, Image0), 2)

    print SpectralImage.shape

    train_x = np.array([])
    train_x.shape = 0, 9
    train_y = np.array([])
    train_y.shape = 0,
    test_x = np.array([])
    test_x.shape = 0, 9
    test_y = np.array([])
    test_y.shape = 0,

    for k in range(1, 18):
        Rindex, Cindex = np.where(GTImage==k)
        if Rindex.shape[0]==0:
            continue
        index_rand = np.random.permutation(Rindex.shape[0])
        Rindex = Rindex[index_rand]
        Cindex = Cindex[index_rand]
        SelectNum = np.int64(Rindex.shape[0] * SelectPer)
        Image0 = SpectralImage[Rindex[0:SelectNum], Cindex[0:SelectNum], :]
        train_x = np.concatenate((train_x, Image0), 0)
        Image0 = GTImage[Rindex[0:SelectNum], Cindex[0:SelectNum]]
        train_y = np.concatenate((train_y, Image0), 0)
        Image0 = SpectralImage[Rindex[SelectNum:], Cindex[SelectNum:], :]
        test_x = np.concatenate((test_x, Image0), 0)
        Image0 = GTImage[Rindex[SelectNum:], Cindex[SelectNum:]]
        test_y = np.concatenate((test_y, Image0), 0)

    print train_x.shape, train_y.shape, test_x.shape, test_y.shape

    index_rand = np.random.permutation(train_x.shape[0])
    print index_rand
    train_x = train_x[index_rand]
    train_y =train_y[index_rand]
    return train_x, train_y, test_x, test_y

def ClassStatistic(prediction, actual):
    StaNum = 17
    prediction = np.int8(prediction)
    actual = np.int8(actual)
    ConfusionMatrix = np.zeros((StaNum, StaNum), dtype=np.int32)
    for i in range(actual.shape[0]):
        ConfusionMatrix[prediction[i]-1, actual[i]-1] +=1

    OverallAccuracy = np.trace(ConfusionMatrix)*1.0 / np.sum(ConfusionMatrix)

    ClassVal = np.sum(ConfusionMatrix, axis=0)
    ClassRight = np.diag(ConfusionMatrix).copy()
    ClassPer = np.zeros((StaNum, 3))
    ClassPer[:, 0] = np.arange(StaNum) + 1
    ClassPer[:, 1] = ClassVal[:]

    ClassRight[ClassVal == 0] = -1
    ClassVal[ClassVal == 0] = 1
    ClassPer0 = ClassRight*1.0/ClassVal
    AA = np.mean(ClassPer0[ClassPer0 != -1])
    ClassPer[:, 2] = ClassPer0[:]

    Pe = np.matmul(np.sum(ConfusionMatrix, axis=0), np.sum(ConfusionMatrix, 1)) *1.0/ np.sum(ConfusionMatrix)**2
    Kappa = (OverallAccuracy - Pe) / (1 - Pe)
    return OverallAccuracy, AA,  Kappa, ClassPer, ConfusionMatrix

train_x, train_y, test_x, test_y = ReadCity(SpectralPath[0], GTFile[0], 0.3)
print train_x.shape

rf = RandomForestClassifier(criterion="entropy", max_features="sqrt", n_estimators=600, min_samples_leaf=2, n_jobs = -1, oob_score=True)  # 这里使用了默认的参数设置
print rf
rf.fit(train_x, train_y)  # 进行模型的训练
print os.system("free -h")

pre_train_y = rf.predict(train_x)
OA1, AA1, Kappa1, ClassPer1, ConfusionMatrix1 = ClassStatistic(pre_train_y, train_y)


pre_test_y = rf.predict(test_x)
OA2, AA2, Kappa2, ClassPer2, ConfusionMatrix2 = ClassStatistic(pre_test_y, test_y)
print OA2, AA2, Kappa2
print ClassPer2[:, 2]
print ConfusionMatrix2



# OA = np.zeros((11, 2))
# for k in range(2, 53, 5):
# # for k in range(100, 1001, 100):
#     print k,
#     rf = RandomForestClassifier(criterion="entropy", max_features="sqrt", n_estimators=600, min_samples_leaf=k, n_jobs = -1, oob_score=True)  # 这里使用了默认的参数设置
#
#     rf.fit(train_x, train_y)  # 进行模型的训练
#     print os.system("free -h")
#
#     pre_train_y = rf.predict(train_x)
#     OA1, AA1, Kappa1, ClassPer1, ConfusionMatrix1 = ClassStatistic(pre_train_y, train_y)
#     OA[np.int64(k/5), 0] = OA1
#
#     pre_test_y = rf.predict(test_x)
#     OA2, AA2, Kappa2, ClassPer2, ConfusionMatrix2 = ClassStatistic(pre_test_y, test_y)
#     OA[np.int64(k / 5), 1] = OA2
#
# print OA[:, 1]
# t = np.arange(2, 53, 5)
# plt.figure
# plt.plot(t, OA[:, 1])
# plt.xticks(t)
# plt.xlabel("The number of min_samples_leaf")
# plt.ylabel("Right rate(%)")
# plt.show()
