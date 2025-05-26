from PIL import Image
from core.preprocessing.processor import Processor

class ArrowPreprocessor:
    patch_size = 64  # valore di default, può essere modificato da fuori

    @classmethod
    def process(cls, batch):

