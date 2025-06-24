r"""
--> Project Structure:
- Model:      C:\Users\nehaa\Downloads\soccer-reid-project\model\best.pt
- Video:      C:\Users\nehaa\Downloads\soccer-reid-project\video\15sec_input_720p.mp4
- Script:     C:\Users\nehaa\Downloads\soccer-reid-project\track_players.py\Code.py
- Output Log: Console + report saved to C:\Users\nehaa\Downloads\soccer-reid-project\report.txt
- Info Doc:   C:\Users\nehaa\Downloads\soccer-reid-project\README.md
"""
# Importing necessary libraries 

import cv2
from ultralytics import YOLO
from scipy.spatial import distance as dist
import numpy as np

# Centroid Tracker with Re-ID

class CentroidTracker:
    def __init__(self, max_disappeared=30, reid_thresh=60):
        self.nextObjectID = 1
        self.objects = {}  
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.reid_thresh = reid_thresh
        self.log_dict = {} 

    def register(self, centroid, frame_id):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.log_dict[self.nextObjectID] = {'first_seen': frame_id, 'reid_frames': []}
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects, frame_id):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], frame_id)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row][col] > self.reid_thresh:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                if frame_id != self.log_dict[objectID]['first_seen']:
                    if frame_id not in self.log_dict[objectID]['reid_frames']:
                        self.log_dict[objectID]['reid_frames'].append(frame_id)
                usedRows.add(row)
                usedCols.add(col)

            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for col in unusedCols:
                self.register(input_centroids[col], frame_id)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

        return self.objects


# Paths 

model_path = r"C:\Users\nehaa\Downloads\soccer-reid-project\model\best.pt"
video_path = r"C:\Users\nehaa\Downloads\soccer-reid-project\video\15sec_input_720p.mp4"
log_path = r"C:\Users\nehaa\Downloads\soccer-reid-project\report.txt"

# Load Model and Initialize

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
tracker = CentroidTracker()
frame_id = 0

# Processing Video Frame-by-Frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model.predict(source=frame, save=False, conf=0.4)
    boxes = results[0].boxes

    # Boxes labeling 
    player_boxes = []
    for i in range(len(boxes)):
        if int(boxes.cls[i]) == 0:  # class 0 = person
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            player_boxes.append(tuple(xyxy))

    objects = tracker.update(player_boxes, frame_id)

    for (objectID, centroid) in objects.items():
        label = f"Player {objectID}"
        cv2.circle(frame, tuple(centroid), 4, (0, 255, 0), -1)
        cv2.putText(frame, label, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Soccer Player Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Final Log Summary

with open(log_path, "w", encoding="utf-8") as f:
    summary = f"\n📋 Re-identification Summary (Total Unique Players: {len(tracker.log_dict)}):\n"
    print(summary)
    f.write(summary)
    for pid, info in tracker.log_dict.items():
        line = f"🧍 Player {pid} - First seen: Frame {info['first_seen']}"
        if info['reid_frames']:
            line += f", Re-identified at: {', '.join(map(str, info['reid_frames']))}"
        print(line)
        f.write(line + "\n")
