from os import listdir
from typing import List, Dict
import json


def annotate(images: List[str], annotation: str) -> Dict[str, str]:
    json_annotations: Dict[str, str] = {}
    for image in images:
        json_annotations[image] = annotation
    return json_annotations


if __name__ == "__main__":
    fa_val_path = "/dataset/source/fa/val"
    fa_test_path = "/dataset/source/fa/test"
    fa_train_path = "/dataset/source/fa/train"

    fca_test_path = "/dataset/source/fca/test"
    fca_train_path = "/dataset/source/fca/train"

    fcb_val_path = "/dataset/source/fcb/val"
    fcb_test_path = "/dataset/source/fcb/test"
    fcb_train_path = "/dataset/source/fcb/train"

    circuit_path = "/dataset/source/circuit"

    class_path = "/dataset/source/class"

    school_path = "/dataset/source/school"

    bpmn_val_path = "/dataset/source/hdBPMN-icdar2021/val"
    bpmn_test_path = "/dataset/source/hdBPMN-icdar2021/test"
    bpmn_train_path = "/dataset/source/hdBPMN-icdar2021/train"

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

    circuit_file_names = [f for f in listdir(circuit_path)]

    class_file_names = [f for f in listdir(class_path)]

    school_file_names = [f for f in listdir(school_path)]

    circuit_json = annotate(circuit_file_names, 'other')

    class_json = annotate(class_file_names, 'other')

    school_json = annotate(school_file_names, 'other')

    bpmn_val_file_names = [f for f in listdir(bpmn_val_path)]
    bpmn_test_file_names = [f for f in listdir(bpmn_test_path)]
    bpmn_train_file_names = [f for f in listdir(bpmn_train_path)]

    bpmn_val_json = annotate(bpmn_val_file_names, 'other')
    bpmn_test_json = annotate(bpmn_test_file_names, 'other')
    bpmn_train_json = annotate(bpmn_train_file_names, 'other')

    bpmn_json = bpmn_val_json
    bpmn_json.update(bpmn_test_json)
    bpmn_json.update(bpmn_train_json)

    labels_json = fa_json
    labels_json.update(fc_json)
    labels_json.update(circuit_json)
    labels_json.update(class_json)
    labels_json.update(school_json)
    labels_json.update(bpmn_json)

    with open('/dataset/classifier/labels.json', 'w') as labels:
        json.dump(labels_json, labels, separators=(',\n', ': '))