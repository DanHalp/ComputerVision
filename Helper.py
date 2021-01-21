import cv2
import numpy as np
import glob
import ntpath
from scipy.signal import convolve2d, convolve
from scipy.spatial.distance import cdist
# from scipy.ndimage import gaussian_filter

ntpath.basename("a/b/c")
RESOURCE_PATH = "../Resources"


def exit(arg):
    return cv2.waitKey(arg) & 0XFF == ord("d")


def object_full_path(obj_name):
    return glob.glob(RESOURCE_PATH + "/**/{}".format(obj_name), recursive=True)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def read_img(img_name):
    img_path = object_full_path(img_name)
    if img_path:
        return img_path[0], cv2.imread(img_path[0])
    else:
        raise Exception("Image is not found")


def show_img_file(img_name, scale=1):
    img_path, img = read_img(img_name)
    img_title = path_leaf(img_path)
    img = rescale_frame(img, scale)
    show_image(img, img_title)


def show_image(img, title=""):
    cv2.imshow(title, img)
    if exit(0):
        return

def show_multi_images(**kwargs):
    for k, v in kwargs.items():
        cv2.imshow(k, v)
    if exit(0):
        return

def pad(im, ph, pw):

    curr_shape = im.shape
    if len(curr_shape) < 3:
        im = im[:, :, None]
    h, w, c = im.shape
    im_padded = np.zeros((h + 2 * ph, w + 2 * pw, c))
    if ph > 0 and pw > 0:
        im_padded[ph:-ph, pw:-pw, :] = im
    elif ph > 0:
        im_padded[ph:-ph, :, :] = im
    elif pw > 0:
        im_padded[:, pw:-pw, :] = im
    else:
        im_padded = im
    return im_padded.reshape(*im_padded.shape[:2]) if len(curr_shape) < 3 else im_padded


def gaussian_kernel(size, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    return np.exp(-(d ** 2 / (2.0 * sigma ** 2)))


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


def convert2gray(im):
    # new_shape = im.shape[:2]
    # conv_kernel = np.array([0.2126, 0.7152, 0.0722])
    # im = im.transpose([2, 0, 1]).reshape(3, -1) / 255
    # im = np.dot(conv_kernel, im).reshape(new_shape)
    # cv2.imshow("grey", im)
    # if exit(0):
    #     return
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def rescale_frame(frame, scale=0.5):
    new_shape = tuple(reversed((scale * np.array(frame.shape)[:2]).astype(int)))
    return cv2.resize(frame, new_shape, interpolation=cv2.INTER_NEAREST)



def draw_contours_by_mask(image, mask, outline=True, rectangle=False):

    canny = canny_edge_detection(mask)
    contours, hierchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1:
            if outline:
                y, x = np.mean(c, axis=0).astype(int)[0]
                r = image[(x, y)]
                if np.mean(r):
                    center, radius = cv2.minEnclosingCircle(c)
                    y, x = np.array(center).astype(int)
                    cv2.circle(image, (y,x), int(radius), (0, 0, 255), thickness=2)
                    print(mask[(x, y)])
                    cv2.drawContours(image, c, -1, (0, 0, 255), thickness=2)
            if rectangle:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    return image

def find_max_colors(image, k):

    indices = np.arange(image.shape[0])
    chosen_pixels = []
    for _ in range(10):
        chosen_pixels = image[np.random.choice(indices, k)]
        dis = cdist([chosen_pixels[0]], chosen_pixels)
        if len(np.unique(dis)) == k:
            break
    unique = np.array(np.unique(chosen_pixels, axis=0))
    return len(unique), unique


def k_means_manually(img, k=2, eps=0.01):

    image = img.reshape((-1, 3)).copy()
    k, means = find_max_colors(image, k)
    labels = []
    old_mean = np.inf
    num_iter = 0
    while np.abs(old_mean - np.mean(means)) > eps or num_iter < 50:
        old_mean = np.mean(means)
        distance = cdist(image, means)
        labels = np.argmin(distance, axis=1)
        means = np.array([np.mean(image[labels == i], axis=0) for i in range(k)])
        num_iter += 1

    image[np.arange(image.shape[0])] = means[labels]
    image = image.reshape(img.shape).astype("uint8")
    return image

def apply_filter_test(image_to_filter, name="tracker"):
    if cv2.getTrackbarPos('1_low', name) == -1:
        create_color_trackbar(name)

    hl = cv2.getTrackbarPos('1_low', name)
    sl = cv2.getTrackbarPos('2_low', name)
    vl = cv2.getTrackbarPos('3_low', name)
    hh = cv2.getTrackbarPos('1_high', name)
    sh = cv2.getTrackbarPos('2_high', name)
    vh = cv2.getTrackbarPos('3_high', name)

    mask = cv2.inRange(image_to_filter, np.array((hl, sl, vl)), np.array((hh, sh, vh)))
    kernel = np.ones((5, 5), "int")
    dilated = cv2.dilate(mask, kernel)
    filtered = cv2.bitwise_and(image_to_filter, image_to_filter, mask=dilated)
    cv2.imshow(name, filtered)
    return filtered

def k_means(image, k=2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    n_img = image.reshape((-1, 3)).astype(np.float32)
    ret, label, center = cv2.kmeans(n_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]

    return res.reshape((image.shape))

def apply_filter(image_to_filter, name="tracker"):
    if cv2.getTrackbarPos('1_low', name) == -1:
        create_color_trackbar(name)

    hl, sl, vl, hh, sh, vh = get_color_trackbar_values(name)
    mask = cv2.inRange(image_to_filter, np.array((hl, sl, vl)), np.array((hh, sh, vh)))
    filtered = cv2.bitwise_and(image_to_filter, image_to_filter, mask=mask)
    cv2.imshow(name, filtered)
    return filtered

def filter_g_b_r(img, window_name="tracker"):

    b_img = gaussian_blur(img, 11)
    image_to_filter = cv2.cvtColor(b_img, cv2.COLOR_BGR2HSV)

    mask = np.zeros_like(image_to_filter)
    iterations = 0
    for i in range(3):
        create_color_trackbar(window_name)
        while True:
            temp = iterations / 5
            if temp == int(temp):
                img_copy = img.copy()
                filtered = apply_filter(image_to_filter, window_name)
                # filtered = apply_filter(img, window_name)
                draw_contours_by_mask(img_copy, filtered)
                cv2.imshow("Original Image", img_copy)
                cv2.imshow(window_name, filtered)
                if (cv2.waitKey(1) & 0XFF == ord("d")):
                    break
            iterations += 1

        mask = cv2.bitwise_or(mask, filtered)

    draw_contours_by_mask(img, mask)
    show_image(img)


def filter_image(img, window_name="tracker"):

    # Create the image to set mask on:
    image_to_filter = k_means_manually(img, 11)
    # b_img = gaussian_blur(img, 11)
    # image_to_filter = cv2.cvtColor(b_img, cv2.COLOR_BGR2HSV)
    while True:
        img_copy = img.copy()
        filtered = apply_filter_test(image_to_filter, window_name)
        # filtered = apply_filter(img, window_name)
        draw_contours_by_mask(img_copy, filtered)
        cv2.imshow("Original Image", img_copy)
        cv2.imshow(window_name, filtered)
        if (cv2.waitKey(1) & 0XFF == ord("d")):
            break


def read_video(vid_name):
    # Notes:
    # 1) VideoCapture(0) reads from the webcam!
    # 2) reading videos is done in a loop.
    vid_path = object_full_path(vid_name)
    if vid_path:
        vid_name = path_leaf(vid_path[0])
        return vid_name, cv2.VideoCapture(vid_path[0])

    else:
        raise Exception("Video is not found")


def play_video(vid, vid_title="", mode=None, vid_scale=1):

    while True:
        isTrue, frame = vid.read()

        # Run a mini neural-network kinda thing.
        # (It just means that the output of one layer is the input of the next one).
        # Note that all "f's" in "nn" take 1 positional argument, which is 'frame'.

        if mode == "filter":
            # to_filter = cv2.cvtColor(k_means_manually(frame, 7), cv2.COLOR_BGR2HSV)
            b_frame = gaussian_blur(frame, 7)
            # to_filter = k_means(cv2.cvtColor(b_frame, cv2.COLOR_BGR2HSV), 5)
            to_filter = cv2.cvtColor(b_frame, cv2.COLOR_BGR2HSV)
            # to_filter = cv2.cvtColor(k_means(frame, 7), cv2.COLOR_BGR2HSV)
            filtered = apply_filter(to_filter)
            draw_contours_by_mask(frame, filtered)

        if (cv2.waitKey(20) & 0XFF == ord("d")) or not isTrue:
            break
        cv2.imshow(vid_title, rescale_frame(frame, vid_scale))

def create_color_trackbar(name="tracker"):
    def nothing(x):
        pass
    cv2.namedWindow(name)
    cv2.createTrackbar('1_low', name, 0, 255, nothing)
    cv2.createTrackbar('2_low', name, 0, 255, nothing)
    cv2.createTrackbar('3_low', name, 0, 255, nothing)
    cv2.createTrackbar('1_high', name, 0, 255, nothing)
    cv2.createTrackbar('2_high', name, 0, 255, nothing)
    cv2.createTrackbar('3_high', name, 0, 255, nothing)
    cv2.setTrackbarPos("1_high", name, 255)
    cv2.setTrackbarPos("2_high", name, 255)
    cv2.setTrackbarPos("3_high", name, 255)
    return name

def get_color_trackbar_values(name="tracker"):
    hl = cv2.getTrackbarPos('1_low', name)
    sl = cv2.getTrackbarPos('2_low', name)
    vl = cv2.getTrackbarPos('3_low', name)
    hh = cv2.getTrackbarPos('1_high', name)
    sh = cv2.getTrackbarPos('2_high', name)
    vh = cv2.getTrackbarPos('3_high', name)
    return hl, sl, vl, hh, sh, vh



if __name__ == '__main__':


    # cap = cv2.VideoCapture(0)
    # play_video(cap, "webcam", mode="filter")
    _, im = read_img("lady.jpg")
    # x, y, _ = im.shape
    # back = np.zeros(im.shape).astype("uint8")
    # cv2.circle(back, (int(x // 2), int(y // 2)), 100, (100, 255, 100), -1)
    # derive_sobel(im)
    # filter_image(im)
    # img = k_means_manually(im, 7)
    # show_image(img)
    filter_image(im)


