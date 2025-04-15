from core.image.image import Image
from dataclasses import dataclass
from torch import Tensor


@dataclass
class TensorImage(Image):
    tensor: Tensor


    @staticmethod
    def from_str(path: str) -> 'Image':
        return TensorImage(path)        # TODO

    def as_tensor(self) -> Tensor:
        return self.tensor
