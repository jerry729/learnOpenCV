import cv2 as cv
import  numpy as np

path = r'imageProcessing/images/noisy.png'
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
blur = cv.GaussianBlur(img, (5,5), 0)

hist = cv.calcHist([blur], [0], None, [256], [0,256]) 

# 当输入图像为多通道图像时，channels[0]={0}，表示的是取其第一个通道，channels[0]={1}表示取其第二个通道，以此类推，channels[2]={1，2}表示的是取其第2个和第3个通道的图像进行直方图统计。

#         mask。掩码。如果mask不为空，那么它必须是一个8位（CV_8U）的数组，并且它的大小的和arrays[i]的大小相同，值为1的点将用来计算

# 直方图。

#         hist。计算出来的直方图

#         dims。计算出来的直方图的维数。

#         histSize。在每一维上直方图的个数。简单把直方图看作一个一个的竖条的话，就是每一维上竖条的个数。

#         ranges。用来进行统计的范围。

# hist是一个一列矩阵， 每一行代表一个直方块的数量
hist_norm = hist.ravel() / hist.sum() # 每一个灰度的概率
Q = hist_norm.cumsum() #累积概率密度

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in range(256):
    p1, p2 = np.hsplit(hist_norm, [i]) #Split an array into multiple sub-arrays horizontally (column-wise). 沿着axis=1分割成sections个 当传入是数组时，表示分割线索引是i，i包括在后面
    q1, q2 = Q[i], Q[255] - Q[i] # cum sum of classes #类概率
    if q1 < 1.e-6 or q2 < 1.e-6:
        continue
    b1, b2 = np.hsplit(bins, [i]) # 每一个灰度值 weights

    # finding means and variances
    m1, m2 = np.sum(b1*p1/q1), np.sum(b2*p2/q2) #类均值
    v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2 # 类内方差

    #使类内方差最小化
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

#用opencv找到的otsu threshold
ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
#和自己找到的作比较
print(f"{thresh}  {ret}")

