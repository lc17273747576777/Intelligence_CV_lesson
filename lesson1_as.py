import cv2
import random
import numpy as np


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

def random_light_color(img, margin):
    # brightness
    B, G, R = cv2.split(img)

    b_rand = random.randint(-margin, margin)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-margin, margin)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-margin, margin)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge


def imagedata_augment(image, fangfeiziwo=0, gamma=1, random_light_color_margin=0, rotate=0, scaled=1,
                      affinea=np.float32([[0, 0], [0, 0], [0, 0]]), affineb=np.float32([[0, 0], [0, 0], [0, 0]]),
                      pers_transforma=np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]),
                      pers_transformb=np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])):
    if fangfeiziwo == 1:
        gamma =1+random.randint(-50, 50)/100.0
        random_light_color_margin= random.randint(0, 60)
        rotate=random.randint(0, 40)
        scaled =1+random.randint(-50, 0)/100.0

    if gamma != 1:
        image = adjust_gamma(image, gamma)
    if random_light_color != 0:
        image = random_light_color(image,random_light_color_margin)
    if rotate != 0 or scaled != 1:
        M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), rotate, scaled)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    if np.linalg.norm(np.subtract(affinea, affineb).flatten())!=0:
        M_a = cv2.getAffineTransform(affinea, affineb)
        image = cv2.warpAffine(image, M_a, (image.shape[1], image.shape[0]))
    if np.linalg.norm(np.subtract(pers_transforma, pers_transformb).flatten())!=0:
        M_t = cv2.getPerspectiveTransform(pers_transforma, pers_transformb)
        image = cv2.warpPerspective(image, M_t, (image.shape[1], image.shape[0]))
    return image


img = cv2.imread('lena_big.jpg')
cv2.imshow('img_origin', img)

img_random_augment = imagedata_augment(img, fangfeiziwo=1)
cv2.imshow('random_augment_pic', img_random_augment)

img_customized_augment = imagedata_augment(img, fangfeiziwo=0, gamma=1.2, random_light_color_margin=30, rotate=15, scaled=0.8)
cv2.imshow('customized_augment_pic', img_customized_augment)

rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
img_affine_augment = imagedata_augment(img, affinea=pts1, affineb=pts2)
cv2.imshow('affine_augment_pic', img_affine_augment)


height, width, channels = img.shape
random_margin = 60
x1 = random.randint(-random_margin, random_margin)
y1 = random.randint(-random_margin, random_margin)
x2 = random.randint(width - random_margin - 1, width - 1)
y2 = random.randint(-random_margin, random_margin)
x3 = random.randint(width - random_margin - 1, width - 1)
y3 = random.randint(height - random_margin - 1, height - 1)
x4 = random.randint(-random_margin, random_margin)
y4 = random.randint(height - random_margin - 1, height - 1)

dx1 = random.randint(-random_margin, random_margin)
dy1 = random.randint(-random_margin, random_margin)
dx2 = random.randint(width - random_margin - 1, width - 1)
dy2 = random.randint(-random_margin, random_margin)
dx3 = random.randint(width - random_margin - 1, width - 1)
dy3 = random.randint(height - random_margin - 1, height - 1)
dx4 = random.randint(-random_margin, random_margin)
dy4 = random.randint(height - random_margin - 1, height - 1)

pts_per1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
pts_per2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
img_perspective_transformed_augment = imagedata_augment(img, pers_transforma=pts_per1, pers_transformb=pts_per2)
cv2.imshow('perspective_transformed_augment_pic', img_perspective_transformed_augment)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
