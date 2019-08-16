import cv2
import numpy as np


# average pooling
def average_pooling(img, G=8):
    out = img.copy()

    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)

    # オリジナル解答
    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int)
    '''
    numpy分かってる感を出す別解
    numpy.mean()は平均をとる軸の番号をtupleで指定可能。
    この例では第0, 1軸での平均を取っている。出力は第2軸の要素数のベクトル(チャネル毎の平均)になる。
    for y in range(Nh):
        for x in range(Nw):
                out[G*y:G*(y+1), G*x:G*(x+1)] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1)], axis=(0,1)).astype(np.int)
    return out
    '''

# Read image
img = cv2.imread("imori.jpg")

# Average Pooling
out = average_pooling(img)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
