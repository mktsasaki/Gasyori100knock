import cv2
import numpy as np
import matplotlib.pyplot as plt


# Nereset Neighbor interpolation
def nn_interpolate(img, ax=1, ay=1):
	H, W, C = img.shape

	aH = int(ay * H)
	aW = int(ax * W)

	# numpy.arange(N) : 0から1ずつインクリメントされた長さNの1次元配列を作成 
	# numpy.repeat(M) : 各要素をM回ずつ繰り返した要素を作成。
	# numpy.reshape() : データの次元を切り直す(データの並びは変わらない。自明な部分は-1で指定可能。
	# numpy.tile(a, (h, w)) : データaを(h, w)個並べる(a が1次元の長さlの配列の場合は aは(1, l)の配列として扱う)

	# 出力サイズでの各画素の座標リストを作成
	y = np.arange(aH).repeat(aW).reshape(-1, aW) # columun数がawになるようにraw数は決定される。→(aH, aW)
	x = np.tile(np.arange(aW), (aH, 1)) # (1, aW)のデータを(aH, 1)個並べる→(aH, aW)個になる
	# 出力画像上の座標が入力画像上のどの位置を参照するかをスケーリングして計算
	y = np.round(y / ay).astype(np.int)
	x = np.round(x / ax).astype(np.int)

	out = img[y,x]

	out = out.astype(np.uint8)

	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float)

# Nearest Neighbor
out = nn_interpolate(img, ax=1.5, ay=1.5)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
