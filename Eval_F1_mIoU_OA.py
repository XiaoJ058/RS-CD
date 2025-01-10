import os
import cv2
import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric  
P\L    P     N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    pred_path = r''
    label_path = r""
    pred_list = []
    pred_file_listdir = os.listdir(pred_path)
    for item in pred_file_listdir:
        file = os.path.join(pred_path, item)
        pred_list.append(cv2.imread(file, -1))
    imgPredict = np.array(pred_list)

    label_list = []
    label_file_listdir = os.listdir(label_path)
    for item_ in label_file_listdir:
        file_ = os.path.join(label_path, item_)
        print(file_)
        label_list.append(cv2.imread(file_, -1))
    imgLabel = np.array(label_list)

    print(imgLabel.shape)
    print(imgPredict.shape)
    imgLabel[imgLabel > 0] = 1
    imgPredict[imgPredict > 0] = 1

    metric = SegmentationMetric(2)
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()

    print("\n\r")
    out = metric.genConfusionMatrix(imgPredict, imgLabel)
    print("metric")
    print("            unchanged     changed")
    print("unchanged   TP-out[0][0]  FP-out[1][0]")
    print("changed     FN-out[0][1]  TN-out[1][1]")
    print("          ", out[0][0], "      ", out[1][0])
    print("          ", out[0][1], "      ", out[1][1])
    print("\n\r")
    print('pa(accuracy) is : %f' % pa)
    print("\n\r")
    print("recall of change area：%.4f" % cpa[1])
    print("\n\r")
    print("pre of unchange area", out[0][0] / (out[0][0] + out[1][0]))
    print("pre of change area",  out[1][1] / (out[0][1] + out[1][1]))
    print("\n\r")
    precision = out[1][1] / (out[0][1] + out[1][1])
    recall = cpa[1]
    print("F1 of change area：", 2 * (precision * recall) / (precision + recall))
    print("\n\r")
    print('mIoU is : %f' % mIoU)

    print("F1", 2 / (1 / precision + 1 / recall))
