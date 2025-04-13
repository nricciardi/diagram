import unittest
from src.dataset.classifier.dataset import ClassifierDataset


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
        self.assertEqual(label, 'graph')


if __name__ == '__main__':
    unittest.main()
