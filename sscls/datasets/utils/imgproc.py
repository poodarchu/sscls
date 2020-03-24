import cv2
import numpy as np


def jpeg_encode(img, quality=90):
    '''
    :param img: uint8 color image array
    :type img: :class:`numpy.ndarray`
    :param int quality: quality for JPEG compression
    :return: encoded image data
    '''
    return cv2.imencode('.jpg', img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tostring()


def png_encode(input, compress_level=3):
    '''
    :param numpy.ndarray input: uint8 color image array
    :param int compress_level: compress level for PNG compression
    :return: encoded image data
    '''
    assert len(input.shape) == 3 and input.shape[2] in [3, 4]
    assert input.dtype == np.uint8
    assert isinstance(compress_level, int) and 0 <= compress_level <= 9
    enc = cv2.imencode('.png', input,
                       [int(cv2.IMWRITE_PNG_COMPRESSION), compress_level])
    return enc[1].tostring()


def imdecode(data, *, require_chl3=True, require_alpha=False):
    """decode images in common formats (jpg, png, etc.)

    :param data: encoded image data
    :type data: :class:`bytes`
    :param require_chl3: whether to convert gray image to 3-channel BGR image
    :param require_alpha: whether to add alpha channel to BGR image

    :rtype: :class:`numpy.ndarray`
    """
    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)

    if img is None and len(data) >= 3 and data[:3] == b'GIF':
        # cv2 doesn't support GIF, try PIL
        img = _gif_decode(data)

    assert img is not None, 'failed to decode'
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1,))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img


def _gif_decode(data):
    try:
        import io
        from PIL import Image

        im = Image.open(io.BytesIO(data))
        im = im.convert('RGB')
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return
