import unittest

import torch

from src.dataset.classifier.dataset import ClassifierDataset
import cv2


class TestGNRDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = ClassifierDataset("/Users/saverionapolitano/PycharmProjects/diagram/dataset/classifier/labels"
                                         ".json",
                                         "/Users/saverionapolitano/PycharmProjects/diagram/dataset/classifier"
                                         "/flowchart_graph")

    def test_len(self):
        self.assertEqual(self.dataset.__len__(), 1391)

    def test_get_item(self):
        image, label = self.dataset.__getitem__(0)
        im = cv2.imread(
            '/Users/saverionapolitano/PycharmProjects/diagram/dataset/classifier/flowchart_graph/writer013_fa_007.png', cv2.IMREAD_UNCHANGED)
        tensor_im = torch.from_numpy(im)
        self.assertTrue(torch.equal(tensor_im[torch.newaxis, :, :], image.as_tensor()))
        self.assertEqual(label, 'graph')


if __name__ == '__main__':
    unittest.main()
