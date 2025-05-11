import json
from typing import List, Dict, Tuple


def merge(test_annotations_path: str, train_annotations_path: str, val_annotations_path: str = '') -> Tuple[List, List]:
    with open(test_annotations_path, 'r') as f:
        test_annotations_json: Dict = json.load(f)

    with open(train_annotations_path, 'r') as f:
        train_annotations_json: Dict = json.load(f)

    if val_annotations_path != '':
        with open(val_annotations_path, 'r') as f:
            val_annotations_json: Dict = json.load(f)

    test_images: List = test_annotations_json['images']
    train_images: List = train_annotations_json['images']
    if val_annotations_path != '':
        val_images: List = val_annotations_json['images']

    test_annotations: List = test_annotations_json['annotations']
    train_annotations: List = train_annotations_json['annotations']
    if val_annotations_path != '':
        val_annotations: List = val_annotations_json['annotations']

    next_id: int = len(test_images)
    images = test_images
    annotations = test_annotations
    for image in train_images:
        for annotation in train_annotations:
            if annotation['image_id'] == image['id']:
                annotation['image_id'] = next_id
                annotations.append(annotation)
        image['id'] = next_id
        next_id += 1
        images.append(image)

    if val_annotations_path != '':
        for image in val_images:
            for annotation in val_annotations:
                if annotation['image_id'] == image['id']:
                    annotation['image_id'] = next_id
                    annotations.append(annotation)
            image['id'] = next_id
            next_id += 1
            images.append(image)

    return images, annotations


def main():
    fa_images, fa_annotations = merge('/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/test.json',
                                      '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/train.json',
                                      '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/val.json')
    fca_images, fca_annotations = merge('/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fca/test.json',
                                        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fca/train.json')
    fcb_images, fcb_annotations = merge('/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb/test.json',
                                        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb/train.json',
                                        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb/val.json')

    labels_json: Dict = {}

    images: List = fa_images
    annotations: List = fa_annotations
    for annotation in annotations:
        if annotation['category'] == 'arrow':
            annotation['category_id'] = 4
        if annotation['category'] == 'text':
            annotation['category_id'] = 3
        if annotation['category'] == 'final state':
            annotation['category_id'] = 2
        if annotation['category'] == 'state':
            annotation['category_id'] = 1

    next_id: int = len(images)
    for image in fca_images:
        for annotation in fca_annotations:
            if annotation['category'] == 'arrow':
                annotation['category_id'] = 4
            if annotation['category'] == 'text':
                annotation['category_id'] = 3
            if annotation['category'] == 'connection':
                annotation['category_id'] = 5
            if annotation['category'] == 'data':
                annotation['category_id'] = 6
            if annotation['category'] == 'decision':
                annotation['category_id'] = 7
            if annotation['category'] == 'process':
                annotation['category_id'] = 8
            if annotation['category'] == 'terminator':
                annotation['category_id'] = 9
            if annotation['image_id'] == image['id']:
                annotation['image_id'] = next_id
                annotations.append(annotation)
        image['id'] = next_id
        next_id += 1
        images.append(image)

    next_id: int = len(images)
    for image in fcb_images:
        for annotation in fcb_annotations:
            if annotation['image_id'] == image['id']:
                if annotation['category'] == 'arrow':
                    annotation['category_id'] = 4
                if annotation['category'] == 'text':
                    annotation['category_id'] = 3
                if annotation['category'] == 'connection':
                    annotation['category_id'] = 5
                if annotation['category'] == 'data':
                    annotation['category_id'] = 6
                if annotation['category'] == 'decision':
                    annotation['category_id'] = 7
                if annotation['category'] == 'process':
                    annotation['category_id'] = 8
                if annotation['category'] == 'terminator':
                    annotation['category_id'] = 9
                annotation['image_id'] = next_id
                annotations.append(annotation)
        image['id'] = next_id
        next_id += 1
        images.append(image)

    labels_json['images'] = images
    labels_json['annotations'] = annotations

    with open('/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/labels.json', 'w') as labels:
        json.dump(labels_json, labels, separators=(',\n', ': '))


main()
