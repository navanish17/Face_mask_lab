import xml.etree.ElementTree as ET

def voc_to_yolo(xml_path, img_w, img_h, class_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_labels = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        cls_id = class_map[cls_name]

        bnd = obj.find("bndbox")
        xmin = int(bnd.find("xmin").text)
        ymin = int(bnd.find("ymin").text)
        xmax = int(bnd.find("xmax").text)
        ymax = int(bnd.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        yolo_labels.append(f"{cls_id} {x_center} {y_center} {w} {h}")

    return yolo_labels
