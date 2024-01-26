import cv2
import numpy as np
from numpy.typing import NDArray

Mat = NDArray[np.uint8]

def image_read(path: str, grayscale: bool = False) -> Mat:
    """
    Reads an image provided as a path, returns a numpy array with its values.

    Arguments:
    path -- path to the image
    grayscale -- indicates whether the image should be converted to grayscale, False by default

    Returns:
    the image as a Numpy array. Note that the default order of subchannels in OpenCV is BGR
    """
    if grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return np.moveaxis(cv2.imread(path), 2, 0)

def image_pad(image, padding='reflect'):
    """
    Pads the image provided as input

    Arguments:
    image -- the image data stored as an NDArray
    
    Keyword arguments:
    padding='reflect' -- the mode of padding; consult Numpy's documentation for np.pad() for the available modes
    """

    # Keep in mind that this is expected to work with image_read, which 
    # *rolls the axes so that the first dim is colour...*

    if len(image.shape) == 2:
        pow2 = 2**int(np.max(np.ceil(np.log2(np.array(image.shape)))))

    pow2 = 2**int(np.max(np.ceil(np.log2(np.array(image.shape[:2])))))

    pad_image = lambda x: np.pad(x, ((pow2-x.shape[0])//2, (pow2-x.shape[1])//2), mode=padding)

    if len(image.shape) == 2:
        return pad_image(image)
    
    out = np.stack([pad_image(image[i]) for i in range(image.shape[0])], axis=0)
    print(out.shape)
    return out

def bgr_to_ycrcb(img: np.ndarray, subsampling: bool = True) -> (Mat, Mat, Mat):
    """
    Returns a tuple containing three separate Numpy arrays holding the image data.
    By default subsamping is on, so the arrays would have different shapes, thus
    in order to accommodate that, the three channels are returned in a tuple.

    Arguments:
    img -- a BGR-ordered color image
    subsampling -- indicates whether chroma subsampling should be applied, True by default (applies 4:2:0 subsampling)

    Returns:
    a tuple of the three channels in the order (Y, Cr, Cb). Note that Y has no gamma correction!
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    Y  = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]

    dims = Y.shape

    print("Before subsampling")    
    print(Y.shape, Cr.shape, Cb.shape)
    

    if not subsampling:
        return (Y, Cr, Cb)

    Cr_sub = Cr[::2, ::2]
    Cb_sub = Cb[::2, ::2]

    print("After subsampling")    
    print(Y.shape, Cr_sub.shape, Cb_sub.shape)


    assert Cr_sub.shape != Y.shape # check if the image is indeed subsampled

    return (Y, Cr_sub, Cb_sub)



def ycrcb_to_bgr(Y: Mat, _Cr: Mat, _Cb: Mat) -> Mat:
    """
    Reconstructs the image from given chrominance channels
    """
    if _Cr.shape == Y.shape:
        YCrCb = np.stack([Y, _Cr, _Cb], axis=2)
        
        
    else:
        Cr = np.zeros_like(Y)
        Cb = np.zeros_like(Y)

        sub_size = _Cr.shape
        full_size = Y.shape

        for i in range(sub_size[0]):
            for j in range(sub_size[1]):
                val_Cr = _Cr[i, j]
                val_Cb = _Cb[i, j]
                Cr[2 * i, 2 * j] = val_Cr
                Cb[2 * i, 2 * j] = val_Cb

                subsample_Y_fits = 2 * i + 1 < full_size[0]
                subsample_X_fits = 2 * j + 1 < full_size[1]
                if subsample_Y_fits:
                    Cr[2 * i + 1, 2 * j] = val_Cr
                    Cb[2 * i + 1, 2 * j] = val_Cb
                
                if subsample_X_fits:
                    Cr[2 * i, 2 * j + 1] = val_Cr
                    Cb[2 * i, 2 * j + 1] = val_Cb

                if subsample_X_fits and subsample_Y_fits:
                    Cr[2 * i + 1, 2 * j + 1] = val_Cr
                    Cb[2 * i + 1, 2 * j + 1] = val_Cb
        
        YCrCb = np.stack([Y, Cr, Cb], axis=2)
        print(YCrCb.shape)

    return cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
