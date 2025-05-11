import unittest

import torch

from src.dataset.extractor.dataset import ObjectDetectionDataset
import cv2


class TestObjectDetectionDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = ObjectDetectionDataset("/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/labels.json",
                                         "/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/flow_graph_diagrams")

    def test_len(self):
        self.assertEqual(self.dataset.__len__(), 1391)

    def test_get_item(self):
        image, target = self.dataset.__getitem__(0)
        im = cv2.imread(
            '/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/flow_graph_diagrams/writer018_fa_001.png', cv2.IMREAD_COLOR)
        tensor_im = torch.from_numpy(im)
        self.assertTrue(torch.equal(tensor_im.swapaxes(0, 2).swapaxes(1, 2), image))
        self.assertEqual(target['boxes'].shape[0], 17)


if __name__ == '__main__':
    unittest.main()
