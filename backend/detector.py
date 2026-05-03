import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

ROWS, COLS = 3, 3

ZONE_NAMES = [
    ["A1","A2","A3"],
    ["B1","B2","B3"],
    ["C1","C2","C3"]
]


def check_overlap(box, zone):
    x1,y1,x2,y2 = box
    zx1,zy1,zx2,zy2 = zone

    return not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2)


def process_frame(frame):
    h, w, _ = frame.shape

    zone_h = h // ROWS
    zone_w = w // COLS

    zone_status = [["Empty" for _ in range(COLS)] for _ in range(ROWS)]

    detected_zones = []
    person_count = 0

    results = model(frame, conf=0.5, verbose=False)

    boxes_list = []

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:

                person_count += 1

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                boxes_list.append((x1,y1,x2,y2))

                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

    for i in range(ROWS):
        for j in range(COLS):

            zx1 = j * zone_w
            zy1 = i * zone_h
            zx2 = (j+1) * zone_w
            zy2 = (i+1) * zone_h

            for box in boxes_list:
                if check_overlap(box,(zx1,zy1,zx2,zy2)):

                    zone_status[i][j] = "Occupied"

                    detected_zones.append(ZONE_NAMES[i][j])

                    break

            color = (0,0,255) if zone_status[i][j]=="Occupied" else (0,255,0)

            cv2.rectangle(frame,(zx1,zy1),(zx2,zy2),color,2)

            cv2.putText(
                frame,
                ZONE_NAMES[i][j],
                (zx1+10, zy1+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    occupancy_percent = int(
        (len(detected_zones)/(ROWS*COLS))*100
    )

    return frame, person_count, occupancy_percent, detected_zones