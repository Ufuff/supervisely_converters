import cv2
import json
import os
import numpy as np


def get_area(corners):
    corners = np.asarray(corners)
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return int(area)


def get_bbox(arr_points):
    arr_points = np.asarray(arr_points)
    arr_x = arr_points[:, 0]
    arr_y = arr_points[:, 1]
    x_min = np.min(arr_x)
    x_max = np.max(arr_x)
    y_min = np.min(arr_y)
    y_max = np.max(arr_y)
    x_left = float(x_min)
    y_left = float(y_min)
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    bbox = [x_left, y_left, width, height]
    return bbox


def coco_img_and_ann(dir_img, dir_ann, h_len, dir_out_img):
    tttt = 0
    annotations = []
    images = []
    files = os.listdir(dir_img)
    id_img = 0
    id_ann = 0
    for file in files:
        json_path = dir_ann + "/" + file + ".json"

        with open(json_path, "r") as read_file:
            json_img = json.load(read_file)
            file_name = file
            anns = json_img["objects"]

            if len(anns) == 3:
                tttt += 1

            if h_len == len(anns):

                images.append(
                    {
                        "license": 0,
                        "url": None,
                        "file_name": file_name,
                        "height": json_img["size"]["height"],
                        "width": json_img["size"]["width"],
                        "date_captured": None,
                        "id": id_img,
                    }
                )

                img = cv2.imread(dir_img + "/" + file_name)
                cv2.imwrite(os.path.join(dir_out_img, file_name), img)

                for ann in anns:
                    points = ann["points"]["exterior"]
                    bbox = get_bbox(points)
                    area = get_area(points)

                    if ann["classTitle"] == "LHand":
                        category_id = 0
                    else:
                        # category_id = 1
                        category_id = 0

                    points = np.asarray(points)
                    points = points.reshape(int(points.shape[0] * 2))
                    points = points.tolist()
                    t = 0
                    list_points = []
                    list_points.append(points)
                    annotations.append(
                        {
                            "id": id_ann,
                            "image_id": id_img,
                            "category_id": category_id,
                            "segmentation": list_points,
                            "area": float(area),
                            "bbox": bbox,
                            "iscrowd": 0,
                        }
                    )
                    id_ann += 1
                id_img += 1

    print("3-h errors")
    print(tttt)
    print("count img")
    print(id_img)
    print("count ann")
    print(id_ann)
    print("============")
    return images, annotations


def convert_supervisely_to_coco(dir_img, dir_ann, path_output_json, dir_out_img):
    result_json = []
    info = []
    categories = []
    images = []
    annotations = []
    licenses = []
    year = 2020
    date_created = "2020-12-09 15:28:08.258300"
    info.append(
        {
            "description": None,
            "url": None,
            "version": None,
            "year": year,
            "contributor": None,
            "date_created": date_created,
        }
    )
    licenses.append({"url": None, "id": 0, "name": None})

    images, annotations = coco_img_and_ann(dir_img, dir_ann, 2, dir_out_img)

    type = "instances"
    categories.append({"supercategory": None, "id": 0, "name": "LHand"})
    # categories.append({'supercategory': None,
    #                    'id': 1,
    #                   'name': 'RHand'
    #                    })
    result_json.append(
        {
            "info": info[0],
            "licenses": licenses,
            "images": images,
            "type": type,
            "annotations": annotations,
            "categories": categories,
        }
    )

    with open(path_output_json + "", "w", encoding="utf-8") as outfile:
        json.dump(result_json[0], outfile, indent=4, ensure_ascii=False)


# json_path = 'annotations.json'

# with open(json_path, "r") as read_file:
#   data = json.load(read_file)

images_dir = "data/supervisely_frames/train_full_1/ds/img"
anns_dir = "data/supervisely_frames/train_full_1/ds/ann"

convert_supervisely_to_coco(
    images_dir,
    anns_dir,
    "result_coco/annotations_train_full_2h.json",
    "result_coco/train/img/",
)


images_dir = "data/supervisely_frames/val_full_1/ds/img"
anns_dir = "data/supervisely_frames/val_full_1/ds/ann"

convert_supervisely_to_coco(
    images_dir,
    anns_dir,
    "result_coco/annotations_val_full_2h.json",
    "result_coco/val/img/",
)
