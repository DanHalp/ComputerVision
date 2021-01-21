from Helper import *


def gaussian_blur(im, gaus_size=3, gaus_sigma=4):
    return cv2.GaussianBlur(im, (gaus_size, gaus_size), cv2.BORDER_DEFAULT)

    # old_shape = im.shape
    # if len(old_shape) < 3:
    #     im = im[:, :, None]
    #
    # g = gaussian_kernel(gaus_size, gaus_sigma)
    # kernel = np.repeat(g[None, :, :], im.shape[2], axis=0)
    # stride = 1
    #
    # h, w, c = im.shape
    # ch, kh, kw,  = kernel.shape
    # ph = int((h * (stride - 1) - stride + kh) / 2)
    # pw = int((w * (stride - 1) - stride + kw) / 2)
    # im_p = pad(im, ph, pw)
    #
    # outHeight = (im_p.shape[0] - kh) // stride + 1
    # outWidth = (im_p.shape[1] - kw) // stride + 1
    #
    # im_p = im_p.transpose([2, 0, 1])
    #
    # im_col = np.zeros((c, kh * kw, outHeight * outWidth))
    # cc = - np.array([stride, stride])  # cc -> current corner of filter
    # p = 0
    # for i in range(outHeight):
    #     cc[0] += stride
    #     for j in range(outWidth):
    #         cc[1] += stride
    #         im_filtered = im_p[:, cc[0]: cc[0] + kh, cc[1]: cc[1] + kw]
    #         im_col[:, :, p] = im_filtered.reshape(-1, kw * kh)
    #         p += 1
    #     cc[1] = - stride
    #
    # res = im_col * kernel.reshape((-1, kh * kw, 1))
    # res = res.mean(axis=1)
    # res = res.reshape((-1, outHeight, outWidth))
    # res = res.transpose([1, 2, 0])
    # return res.reshape(*res.shape[:2])if len(old_shape) < 3 else res