import cv2
import numpy as np


# Gaussian filter ( Q9. の別解)
def gaussian_filter(img, K_size=3, sigma=1.3):
	'''
	param:
		img : 入力画像(3plane)
		K_size : Filter Kernelサイズ(奇数)
		sigma : Gaussianのsigma
	returns:
		out : Gaussian filter適用結果の画像
	'''
	H, W, C = img.shape # 入力画像のshapeを取得

	## Zero padding
	pad = K_size // 2
	out = np.zeros_like(img) # imgと同じshape、typeの0埋めarray作成
	# 外周に0 padding した画像の作成
	tmp = np.zeros((H + pad*2, W + pad*2, C), dtype=np.uint8)
	tmp[pad:pad+H, pad:pad+W] = img.copy().astype(np.uint8) # 中央に元画像をコピー

	## prepare Kernel
	K = np.zeros((K_size, K_size), dtype=np.float)
	for x in range(-pad, -pad+K_size):
		for y in range(-pad, -pad+K_size):
			K[y+pad, x+pad] = np.exp( -(x**2 + y**2) / (2* (sigma**2)))
	K /= K.sum() # Kernel全体の値の合計が1.0になるように正規化
	# (H, W)のarrayを(H, W, 1)の形に拡張
	K = np.reshape(K, K.shape + (1,))

	# filtering
	for y in range(H):
		for x in range(W):
			out[y, x] = np.sum(K * tmp[y:y+K_size, x:x+K_size], axis=(0,1)).astype(np.uint8)
	return out


# Read image
img = cv2.imread("imori_noise.jpg")


# Gaussian Filter
out = gaussian_filter(img, K_size=3, sigma=1.3)


# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
