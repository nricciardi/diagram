import os
import json
import argparse
from PIL import Image
import shutil
from tqdm import tqdm

ARROW_CATEGORY = 6

def extract_patch(img, center, patch_size):
    x, y = int(center[0]), int(center[1])
    half = patch_size // 2
    left = max(x - half, 0)
    top = max(y - half, 0)
    right = left + patch_size
    bottom = top + patch_size
    return img.crop((left, top, right, bottom)), (left, top, right, bottom)

def main(coco_json_path, images_dir, output_dir, output_name):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    patch_output = os.path.join(output_dir, output_name)
    os.makedirs(patch_output, exist_ok=True)

    img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    patches_info = []

    for ann in tqdm(coco['annotations']):
        if ann['category_id'] != ARROW_CATEGORY:
            continue

        keypoints = ann.get('keypoints', [])
        if not keypoints or len(keypoints) < 6:
            continue

        tail = keypoints[:2]
        head = keypoints[3:5]

        if head == [0.0, 0.0] or tail == [0.0, 0.0]:
            continue

        image_id = ann['image_id']
        image_filename = img_id_to_file.get(image_id)
        if not image_filename:
            print(f"Image with id {image_id} not found")
            continue

        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found")
            continue

        shutil.copy2(image_path, os.path.join(output_dir, output_name))

        for point, label in zip([head, tail], ["head", "tail"]):
            target_x = point[0]
            target_y = point[1]

            patches_info.append({
                "image": image_filename,
                "label": label,
                "arrow_id": ann['id'],
                "target_x": target_x,
                "target_y": target_y
            })

    output_json = os.path.join(output_dir, f"{output_name}.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(patches_info, f, indent=2)

    print(f"{len(patches_info)} patches generated")
    print(f"Patches saved into {patch_output}")
    print(f"JSON file saved into {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrai patch testa/coda frecce da dataset COCO-like.")
    parser.add_argument("--coco_json", required=True, help="Path al file JSON in formato COCO-like.")
    parser.add_argument("--images_dir", required=True, help="Directory contenente le immagini.")
    parser.add_argument("--output_dir", required=True, help="Directory di output per le patch e il JSON.")
    parser.add_argument("--output_name", required=True, help="Output names.")
    args = parser.parse_args()

    main(args.coco_json, args.images_dir, args.output_dir, args.output_name)
