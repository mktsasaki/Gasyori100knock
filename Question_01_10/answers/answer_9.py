import cv2
import numpy as np


# Gaussian filter
def gaussian_filter(img, K_size=3, sigma=1.3):
	H, W, C = img.shape

	## Zero padding
	pad = K_size // 2
	out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
	out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

	## prepare Kernel
	K = np.zeros((K_size, K_size), dtype=np.float)
	for x in range(-pad, -pad+K_size):
		for y in range(-pad, -pad+K_size):
			K[y+pad, x+pad] = np.exp( -(x**2 + y**2) / (2* (sigma**2)))
	K /= (sigma * np.sqrt(2 * np.pi)) # ⇦この行、ガウス関数の定義上は必要ですが、実は次の行で正規化を行うので不要です。
	K /= K.sum() # Kernel全体の値の合計が1.0になるように正規化

	tmp = out.copy()

	# filtering
	for y in range(H):
		for x in range(W):
			for c in range(C):
				out[pad+y, pad+x, c] = np.sum(K * tmp[y:y+K_size, x:x+K_size, c])

	out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

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
