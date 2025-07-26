import PIL.Image
from PIL import Image
import cv2
import numpy as np
import random
import concurrent.futures
from scipy.ndimage import rotate

def img_to_patches(image) -> tuple:
    """
    Returns 32x32 patches of a resized 256x256 images,
    it returns 64x64 patches on grayscale and 64x64 patches
    on the RGB color scale
    --------------------------------------------------------
    ## parameters:
    - input_path: Accepts input path of the image
    """
    img = Image.fromarray(image.astype('uint8'))
    if(img.size!=(256,256)):
        print("ATTENZIOEN", img.size)
    patch_size = 32
    grayscale_imgs = []
    imgs = []
    for i in range(0,img.height,patch_size):
        for j in range(0, img.width, patch_size):
            box = (j,i,j+patch_size,i+patch_size)
            img_color = np.asarray(img.crop(box))
            grayscale_image = cv2.cvtColor(src=img_color, code=cv2.COLOR_RGB2GRAY)
            grayscale_imgs.append(grayscale_image.astype(dtype=np.int32))
            imgs.append(img_color)
    return grayscale_imgs,imgs



def get_l1(v,x,y):
    l1=0
    # 1 to m, 1 to m-1
    for i in range(0,y-1):
        for j  in range(0,x):
            l1+=abs(v[j][i]-v[j][i+1])
    return l1

def get_l2(v,x,y):
    l2=0
    # 1 to m-1, 1 to m
    for i in range(0,y):
        for j  in range(0,x-1):
            l2+=abs(v[j][i]-v[j+1][i])
    return l2

def get_l3l4(v,x,y):
    l3=l4=0
    # 1 to m-1, 1 to m-1
    for i in range(0,y-1):
        for j  in range(0,x-1):
            l3+=abs(v[j][i]-v[j+1][i+1])
            l4+=abs(v[j+1][i]-v[j][i+1])

    return l3+l4

def get_pixel_var_degree_for_patch(patch:np.array)->int:
    """
    gives pixel variation for a given patch
    ---------------------------------------
    ## parameters:
    - patch: accepts a numpy array format of the patch of an image
    """
    x,y = patch.shape
    l1=l2=l3l4=0

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_l1 = executor.submit(get_l1,patch,x,y)
        future_l2 = executor.submit(get_l2,patch,x,y)
        future_l3l4 = executor.submit(get_l3l4,patch,x,y)

        l1 = future_l1.result()
        l2 = future_l2.result()
        l3l4 = future_l3l4.result()

    return  l1+l2+l3l4


def extract_rich_and_poor_textures(variance_values:list, patches:list):
    """
    returns a list of rich texture and poor texture patches respectively
    --------------------------------------------------------------------
    ## parameters:
    - variance_values: list of values that are pixel variances of each patch
    - color_patches: coloured patches of the target image
    """
    threshold = np.mean(variance_values)
    rich_texture_patches = []
    poor_texture_patches = []
    for i,j in enumerate(variance_values):
        if j >= threshold:
            rich_texture_patches.append(patches[i])
        else:
            poor_texture_patches.append(patches[i])

    return rich_texture_patches, poor_texture_patches



def get_complete_image(patches:list, coloured=True):
    """
    Develops complete 265x256 image from rich and poor texture patches
    ------------------------------------------------------------------
    ## parameters:
    - patches: Takes a list of rich or poor texture patches
    """
    random.shuffle(patches)
    p_len = len(patches)
    while len(patches)<64:
        patches.append(patches[random.randint(0, p_len-1)])

    if(coloured==True):
        grid = np.asarray(patches).reshape((8,8,32,32,3))
    else:
        grid = np.asarray(patches).reshape((8,8,32,32))


    # joins columns to only leave rows
    rows = [np.concatenate(grid[i,:], axis=1) for i in range(8)]

    # joins the rows to create the final image
    img = np.concatenate(rows,axis=0)

    return img



def smash_n_reconstruct(image_and_label, coloured=True):
    """
    Performs the SmashnReconstruct part of preprocesing
    reference: [link](https://arxiv.org/abs/2311.12397)

    return rich_texture,poor_texture

    ----------------------------------------------------
    ## parameters:
    - input_path: Accepts input path of the image
    """
    label = image_and_label[1]
    image = image_and_label[0]
    gray_scale_patches, color_patches = img_to_patches(image)
    pixel_var_degree = []
    for patch in gray_scale_patches:
        pixel_var_degree.append(get_pixel_var_degree_for_patch(patch))

    # r_patch = list of rich texture patches, p_patch = list of poor texture patches
    if(coloured):
        r_patch,p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree,patches=color_patches)
    else:
        r_patch,p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree,patches=gray_scale_patches)
    rich_texture,poor_texture = None,None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rich_texture_future = executor.submit(get_complete_image,r_patch,coloured)
        poor_texture_future = executor.submit(get_complete_image,p_patch,coloured)

        rich_texture = rich_texture_future.result()
        poor_texture = poor_texture_future.result()

    return rich_texture, poor_texture, label

def apply_filter_a(src:np.ndarray):
    src_copy = np.copy(src)
    f1 = np.array([[[ 0,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  1, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]]])

    img = cv2.filter2D(src=src_copy, kernel=f1[0], ddepth=-1)
    for filter in f1[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//8

def apply_filter_b(src:np.ndarray):
    src_copy = np.copy(src)
    f2 = np.array([[[ 0,  0,  0,  0,  0],
                    [ 0,  2,  1,  0,  0],
                    [ 0,  1, -3,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0, -1,  0,  0],
                    [ 0,  0,  3,  0,  0],
                    [ 0,  0, -3,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  2,  0],
                    [ 0,  0, -3,  1,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  1, -3,  3, -1],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0, -3,  1,  0],
                    [ 0,  0,  1,  2,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -3,  0,  0],
                    [ 0,  0,  3,  0,  0],
                    [ 0,  0, -1,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  1, -3,  0,  0],
                    [ 0,  2,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [-1,  3, -3,  1,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]])

    img = cv2.filter2D(src=src_copy, kernel=f2[0], ddepth=-1)
    for filter in f2[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//8



def apply_filter_c(src:np.ndarray):
    src_copy=np.copy(src)
    f3 = np.array([[[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  1, -2,  1,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]])

    img = cv2.filter2D(src=src_copy, kernel=f3[0], ddepth=-1)
    for filter in f3[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//4


def apply_filter_d(src:np.ndarray):
    src_copy=np.copy(src)
    f4 = np.array([[[ 0,  0,  0,  0,  0],
                    [ 0, -1,  2, -1,  0],
                    [ 0,  2, -4,  2,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0, -1,  2,  0,  0],
                    [ 0,  2, -4,  0,  0],
                    [ 0, -1,  2,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  2, -4,  2,  0],
                    [ 0, -1,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  2, -1,  0],
                    [ 0,  0, -4,  2,  0],
                    [ 0,  0,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]]])

    img = cv2.filter2D(src=src_copy, kernel=f4[0], ddepth=-1)
    for filter in f4[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//4

def apply_filter_e(src:np.ndarray):
    src_copy=np.copy(src)
    f5 = np.array([[[  1,   2,  -2,   2,   1],
                    [  2,  -6,   8,  -6,   2],
                    [ -2,   8, -12,   8,  -2],
                    [  0,   0,   0,   0,   0],
                    [  0,   0,   0,   0,   0]],

                [[  1,   2,  -2,   0,   0],
                    [  2,  -6,   8,   0,   0],
                    [ -2,   8, -12,   0,   0],
                    [  2,  -6,   8,   0,   0],
                    [  1,   2,  -2,   0,   0]],

                [[  0,   0,   0,   0,   0],
                    [  0,   0,   0,   0,   0],
                    [ -2,   8, -12,   8,  -2],
                    [  2,  -6,   8,  -6,   2],
                    [  1,   2,  -2,   2,   1]],

                [[  0,   0,  -2,   2,   1],
                    [  0,   0,   8,  -6,   2],
                    [  0,   0, -12,   8,  -2],
                    [  0,   0,   8,  -6,   2],
                    [  0,   0,  -2,   2,   1]]])

    img = cv2.filter2D(src=src_copy, kernel=f5[0], ddepth=-1)
    for filter in f5[1:]:
        img=cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//4

def apply_filter_f(src:np.ndarray):
    src_copy=np.copy(src)
    f5 = np.asarray([[ 0,  0,  0,  0,  0],
                    [ 0,  -1,  2, -1,  0],
                    [ 0,  2,  -4,  2,  0],
                    [ 0,  -1,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]])

    img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
    return img


def apply_filter_g(src:np.ndarray):
    src_copy=np.copy(src)
    f5 = np.asarray([[ -1,   2,  -2,   2,  -1],
                    [  2,  -6,   8,  -6,   2],
                    [ -2,   8, -12,   8,  -2],
                    [  2,  -6,   8,  -6,   2],
                    [ -1,   2,  -2,   2,  -1]])

    img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
    return img

def apply_all_filters(src:np.ndarray):
    src_copy = np.copy(src)
    img = np.array(cv2.cvtColor((apply_filter_a(src_copy) + apply_filter_b(src_copy) + apply_filter_c(src_copy) + \
            apply_filter_d(src_copy) + apply_filter_e(src_copy) + apply_filter_f(src_copy) + apply_filter_g(src_copy)), cv2.COLOR_RGB2GRAY)//7)
    img_thresh = np.median(img)+2
    return cv2.threshold(img,img_thresh,255,cv2.THRESH_BINARY)[1]












