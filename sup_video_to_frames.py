import cv2
import json
import os


def get_frames_from_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()

    while ret:
        frames.append(image)
        ret, image = cap.read()
    return frames


def get_ds_names(folder_name):
    try:
        list = os.listdir(os.path.join("data", folder_name))
        paths = []
        for name in list:
            cur_path = os.path.join("data", folder_name, name)
            if os.path.isdir(cur_path):
                paths.append(cur_path)
        return paths
    except:
        print("Folder not exist")


def parse_video_to_img(video_name, id):
    # try:
    json_path = "data/test_rsl/ds0/ann/" + video_name + ".json"
    video_path = "data/test_rsl/ds0/video/" + video_name
    number = 0
    meta = []
    classes = []
    temp = []
    arr_ann = []
    # fileName = os.path.basename(os.path.realpath(video_name))
    fileName = os.path.splitext(video_name)[0]
    with open(json_path, "r") as read_file:
        print(read_file)
        video_frames = get_frames_from_video(video_path)
        data = json.load(read_file)

        size = data["size"]
        tags = data["tags"]
        description = data["description"]

        objects = data["objects"]
        objects_dict = {}
        for object in objects:
            key = object["key"]
            label = object["classTitle"]
            objects_dict[key] = label

        classes.append(
            {
                "title": "LHand",
                "shape": "bitmap",
                "color": "#B8E986",
                "geometry_config": {},
                "id": 1724412,
                "hotkey": "L",
            }
        )

        classes.append(
            {
                "title": "RHand",
                "shape": "bitmap",
                "color": "#DD1866",
                "geometry_config": {},
                "id": 1723985,
                "hotkey": "R",
            }
        )

        meta.append({"classes": classes, "tags": [], "projectType": "images"})

        with open("result_parse/meta.json", "w", encoding="utf-8") as outfile:
            json.dump(meta[0], outfile, indent=4, ensure_ascii=False)

        frames = data["frames"]
        for frame in frames:
            head = []
            obj = []
            index = frame["index"]
            figures = frame["figures"]
            for figure in figures:

                base64_data = figure["geometry"]["bitmap"]["data"]
                mask_full = base64_data

                bitmap = figure["geometry"]["bitmap"]

                geometryType = figure["geometryType"]

                if objects[0]["key"] == figure["objectKey"]:
                    classId = 1723985
                    classTitle = "RHand"
                else:
                    classId = 1724412
                    classTitle = "LHand"
                id += 1
                obj.append(
                    {
                        "id": id,
                        "classId": classId,
                        "description": description,
                        "geometryType": geometryType,
                        "tags": tags,
                        "classTitle": classTitle,
                        "bitmap": bitmap,
                    }
                )
            head.append(
                {"description": description, "tags": tags, "size": size, "objects": obj}
            )
            with open(
                "result_parse/ds/ann/"
                + fileName
                + "_"
                + str(number)
                + ".jpg"
                + ".json",
                "w",
                encoding="utf-8",
            ) as outfile:
                json.dump(head[0], outfile, indent=4, ensure_ascii=False)

            frame_vid = video_frames[index]
            # frame_vid = cv2.rotate(frame_vid, cv2.ROTATE_180)
            cv2.imwrite(
                os.path.join(
                    "result_parse/ds/img/", fileName + "_" + str(number) + ".jpg"
                ),
                frame_vid,
            )
            number += 1
    return id


id = 509104798
dir = "data/test_rsl/ds0/video"
files = os.listdir(dir)
for file in files:
    id = parse_video_to_img(os.path.basename(os.path.realpath(file)), id)
print(id)
