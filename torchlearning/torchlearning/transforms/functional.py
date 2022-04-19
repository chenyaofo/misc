from PIL import Image
import io


def bytes2image(bytes, mode):
    return Image.open(io.BytesIO(bytes)).convert(mode=mode)
