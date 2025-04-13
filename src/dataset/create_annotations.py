from os import listdir
from typing import List, Dict
import json


def annotate(images: List[str], annotation: str) -> Dict[str, str]:
    json_annotations: Dict[str, str] = {}
    for image in images:
        json_annotations[image] = annotation
    return json_annotations


if __name__ == "__main__":
    fa_val_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/val"
    fa_test_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/test"
    fa_train_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/train"

    fca_test_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fca/test"
    fca_train_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fca/train"

    fcb_val_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb/val"
    fcb_test_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb/test"
    fcb_train_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb/train"

    fcb_scan_val_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb_scan/val"
    fcb_scan_test_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb_scan/test"
    fcb_scan_train_path = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb_scan/train"

    fa_val_file_names = [f for f in listdir(fa_val_path)]
    fa_test_file_names = [f for f in listdir(fa_test_path)]
    fa_train_file_names = [f for f in listdir(fa_train_path)]

    fa_json = annotate(fa_val_file_names, 'graph')
    fa_test_json = annotate(fa_test_file_names, 'graph')
    fa_train_json = annotate(fa_train_file_names, 'graph')

    fa_json.update(fa_test_json)
    fa_json.update(fa_train_json)

    fca_test_file_names = [f for f in listdir(fca_test_path)]
    fca_train_file_names = [f for f in listdir(fca_train_path)]

    fca_json = annotate(fca_test_file_names, 'flowchart')
    fca_train_json = annotate(fca_train_file_names, 'flowchart')

    fca_json.update(fca_train_json)

    fcb_val_file_names = [f for f in listdir(fcb_val_path)]
    fcb_test_file_names = [f for f in listdir(fcb_test_path)]
    fcb_train_file_names = [f for f in listdir(fcb_train_path)]

    fcb_json = annotate(fcb_val_file_names, 'flowchart')
    fcb_test_json = annotate(fcb_test_file_names, 'flowchart')
    fcb_train_json = annotate(fcb_train_file_names, 'flowchart')

    fcb_json.update(fcb_test_json)
    fcb_json.update(fcb_train_json)

    fc_json = fca_json
    fc_json.update(fcb_json)

    labels_json = fa_json
    labels_json.update(fc_json)

    with open('/Users/saverionapolitano/PycharmProjects/diagram/dataset/classifier/labels.json', 'w') as labels:
        json.dump(labels_json, labels, separators=(',\n', ': '))

    fcb_scan_val_file_names = [f for f in listdir(fcb_scan_val_path)]
    fcb_scan_test_file_names = [f for f in listdir(fcb_scan_test_path)]
    fcb_scan_train_file_names = [f for f in listdir(fcb_scan_train_path)]

    fcb_scan_json = annotate(fcb_scan_val_file_names, 'flowchart')
    fcb_scan_test_json = annotate(fcb_scan_test_file_names, 'flowchart')
    fcb_scan_train_json = annotate(fcb_scan_train_file_names, 'flowchart')

    fcb_scan_json.update(fcb_scan_test_json)
    fcb_scan_json.update(fcb_scan_train_json)
