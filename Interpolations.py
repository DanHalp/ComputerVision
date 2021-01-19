import numpy as np
import cv2 as cv


def resize_image(image, f, resoultion=None, intepolation="bilinear"):
    """
    If f < 0, squeeze image by f. Otherwise, enlarge the image by given interpolation arg.
    :param image: Given input
    :param f: by how much to increase the image shape.
    :return:
    """

    def _resize_by_f(img, f, axis=1):

        if f < 1:
            if axis:
                tw = np.linspace(0, W - 1, num=int(W * f)).astype(int)
                return img[:, :, tw]
            th = np.linspace(0, H - 1, num=int(H*f)).astype(int)
            return img[:, th]

        if intepolation == "bilinear":
            if axis:
                return linear_interpolation(img, f)
            # Perform interpolation along axis=0
            return linear_interpolation(img.transpose([0, 2, 1]), f).transpose([0, 2, 1])
        else:
            raise Exception(f"Invalid interpolation arg: {intepolation}")

    grayscale = False
    if image.ndim < 3:
        # Deal with grayscale images
        grayscale = True
        image = image[None]

    _, H, W = image.shape
    if resoultion:
        fh, fw = np.array(resoultion) / np.array([H, W])
    else:
        fh, fw = f

    res = _resize_by_f(image, fh, axis=0)
    res = _resize_by_f(res, fw, axis=1)

    return res[0] if grayscale else res


def linear_interpolation(inp, f):
    """
    Liniar interpolation along axis=1 (width of a vector / matrix)
    :param inp: matrix of shape (C, H, W) where H, W >= 1
    :param f: factor. The number by which the new distance between each pixel is decided (along axis).
    :return: interpolated matrix.
    """
    if f <= 1:
        return inp

    C, H, W = inp.shape
    ind = np.arange(W)
    t = np.linspace(0, W - 1, num=int(W*f))

    # Rates from left, and rates from the right. Debug to understand.
    # Let say f =  2 and W = 2:
    # ind = [0,1]. t = [0, 1/3, 2/3, 1]
    # rl_i = [0, 0, 0, 1], rr_i = [0, 1, 1, 1]
    # rl = [0, 2/3, 1/3, 0], rr = [0, 1/3, 2/3, 0]
    rl_i, rr_i = np.floor(t).astype(int), np.ceil(t).astype(int)
    rr, rl = t - rl_i, rr_i - t
    res = inp[:, :, rl_i] * rl + inp[:, :, rr_i] * rr

    # Put original values into the corresponding indices.
    Hr = res.shape[2]
    new_ind = np.ceil(ind * Hr * (1 / W)).astype(int)
    res[:, :, new_ind] = inp[:, :, ind]
    return res


if __name__ == '__main__':
    img = cv.imread(r"Images\mona_lisa.jpg")
    img = img.transpose([2, 0, 1])
    en = resize_image(img, (0.2,0.3)).astype("uint8")
    img = img.transpose([1, 2, 0])
    en = en.transpose([1, 2, 0])
    cv.imshow("new", en)
    cv.imshow("org", img)
    cv.waitKey(0)

