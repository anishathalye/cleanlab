from typing import TypeVar, Generic, Tuple, Iterator, Any, List
from typing_extensions import TypeAlias
import numpy.typing as npt
import pandas as pd

T_index = TypeVar("T_index", contravariant=True)
T_data = TypeVar("T_data", covariant=True)
T_label = TypeVar("T_label", covariant=True)


class Dataset(Generic[T_index, T_data]):
    def __getitem__(self, index: T_index) -> T_data:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[T_index, T_data]]:
        raise NotImplementedError


LabeledDataset: TypeAlias = Dataset[T_index, Tuple[T_data, T_label]]


# users are allowed to subclass Dataset, but some of our tools will require
# subclasses of specific classes (e.g., something that analyzes images might
# only handle subclasses of ImageDataset)

# design decision: images are represented (externally) as np.ndarrays
#
# XXX how can we naturally support both labeled and unlabeled image datasets?
class ImageDataset(Dataset[T_index, Tuple[npt.NDArray[Any], T_label]]):
    pass  # also an abstract class


# an example of modality-specific features we might support
def find_blurry_images(ds: ImageDataset[T_index, T_label]) -> List[int]:
    raise NotImplementedError  # TODO


class Analyzer:
    def __init__(self, dataset: Dataset[T_index, T_data]):
        self._dataset = dataset

    # XXX what kinds of functionality do we want to support here


# an example of some concrete dataset classes


class NumpyImageDataset(ImageDataset[int, T_label]):
    def __init__(self, images: npt.NDArray[Any], labels: List[T_label]):
        self._images = images
        self._labels = labels

    def __getitem__(self, index: int) -> Tuple[npt.NDArray[Any], T_label]:
        return self._images[index], self._labels[index]

    def __iter__(self) -> Iterator[Tuple[int, Tuple[npt.NDArray[Any], T_label]]]:
        raise NotImplementedError  # TODO


# this class wouldn't be in the cleanlab open source, it'll probably just be in
# cleanlab-studio-training
#
# example: we made the choice to make the labels a "str" here
class S3ImageDataset(ImageDataset[str, str]):
    def __init__(self, paths: List[str], labels: List[str]):
        self._paths = paths
        self._labels = labels

    def __getitem__(self, index: str) -> Tuple[npt.NDArray[Any], str]:
        # fetch from S3
        raise NotImplementedError  # TODO


class TabularDataset(LabeledDataset[int, pd.Series[Any], Any]):
    def __init__(self, df: pd.DataFrame, label_column: Any):
        raise NotImplementedError  # TODO

    @classmethod
    def from_csv(cls, path: str) -> TabularDataset:
        raise NotImplementedError  # TODO

    @classmethod
    def from_xlsx(cls, path: str) -> TabularDataset:
        raise NotImplementedError  # TODO
