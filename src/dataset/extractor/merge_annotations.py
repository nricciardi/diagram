import json
from typing import List, Dict, Tuple
import copy

def merge(test_annotations_path: str, train_annotations_path: str, id_start: int, id_annotation_start: int,
          val_annotations_path: str = '') -> Tuple[List, List, int, int]:
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

    next_id: int = id_start if id_start > 0 else len(test_images)
    next_annotation_id: int = id_annotation_start if id_annotation_start > 0 else len(test_annotations)
    images = test_images
    annotations = test_annotations
    im_cp: List = copy.deepcopy(train_images)
    ann_cp: List = copy.deepcopy(train_annotations)
    for image, im in zip(train_images, im_cp):
        for annotation, ann in zip(train_annotations, ann_cp):
            if annotation['image_id'] == image['id']:
                ann['id'] = next_annotation_id
                next_annotation_id += 1
                ann['image_id'] = next_id
                annotations.append(ann)
        im['id'] = next_id
        next_id += 1
        images.append(im)

    if val_annotations_path != '':
        im_cp = copy.deepcopy(val_images)
        ann_cp = copy.deepcopy(val_annotations)
        for image, im in zip(val_images, im_cp):
            for annotation, ann in zip(val_annotations, ann_cp):
                if annotation['image_id'] == image['id']:
                    ann['id'] = next_annotation_id
                    next_annotation_id += 1
                    ann['image_id'] = next_id
                    annotations.append(ann)
            im['id'] = next_id
            next_id += 1
            images.append(im)

    return images, annotations, next_id, next_annotation_id


def main():
    fa_images, fa_annotations, id_im, id_ann = merge(
        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/test.json',
        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/train.json', 0, 0,
        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fa/val.json')
    fca_images, fca_annotations, id_im, id_ann = merge(
        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fca/test.json',
        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fca/train.json', id_im, id_ann)
    fcb_images, fcb_annotations, id_im, id_ann = merge(
        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb/test.json',
        '/Users/saverionapolitano/PycharmProjects/diagram/dataset/source/fcb/train.json', id_im, id_ann,
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
    next_ann_id: int = len(annotations)
    im_cp = copy.deepcopy(fca_images)
    ann_cp = copy.deepcopy(fca_annotations)
    for image, im in zip(fca_images, im_cp):
        for annotation, ann in zip(fca_annotations, ann_cp):
            if annotation['category'] == 'arrow':
                ann['category_id'] = 4
            if annotation['category'] == 'text':
                ann['category_id'] = 3
            if annotation['category'] == 'connection':
                ann['category_id'] = 5
            if annotation['category'] == 'data':
                ann['category_id'] = 6
            if annotation['category'] == 'decision':
                ann['category_id'] = 7
            if annotation['category'] == 'process':
                ann['category_id'] = 8
            if annotation['category'] == 'terminator':
                ann['category_id'] = 9
            if annotation['image_id'] == image['id']:
                ann['image_id'] = next_id
                ann['id'] = next_ann_id
                next_ann_id += 1
                annotations.append(ann)
        im['id'] = next_id
        next_id += 1
        images.append(im)

    im_cp = copy.deepcopy(fcb_images)
    ann_cp = copy.deepcopy(fcb_annotations)
    for image, im in zip(fcb_images, im_cp):
        for annotation, ann in zip(fcb_annotations, ann_cp):
            if annotation['image_id'] == image['id']:
                if annotation['category'] == 'arrow':
                    ann['category_id'] = 4
                if annotation['category'] == 'text':
                    ann['category_id'] = 3
                if annotation['category'] == 'connection':
                    ann['category_id'] = 5
                if annotation['category'] == 'data':
                    ann['category_id'] = 6
                if annotation['category'] == 'decision':
                    ann['category_id'] = 7
                if annotation['category'] == 'process':
                    ann['category_id'] = 8
                if annotation['category'] == 'terminator':
                    ann['category_id'] = 9
                ann['image_id'] = next_id
                ann['id'] = next_ann_id
                next_ann_id += 1
                annotations.append(ann)
        im['id'] = next_id
        next_id += 1
        images.append(im)

    labels_json['images'] = images
    labels_json['annotations'] = annotations

    with open('/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/labels.json', 'w') as labels:
        json.dump(labels_json, labels, separators=(',\n', ': '))


main()
