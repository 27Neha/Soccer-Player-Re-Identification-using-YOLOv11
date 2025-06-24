# ⚽ Soccer Player Re-Identification using YOLOv11

Welcome to my internship project — a real-time **soccer player tracking system** using a 15-second match clip. This system is built to detect every player, assign each a **unique ID**, and most importantly, **re-identify players** even if they leave and later return to the frame — just like a human would recognize someone they saw earlier.

---

## 🎯 Objective

The goal of this project is to:
- Detect **all players** in the video using a fine-tuned YOLOv11 object detection model.
- Assign **consistent unique IDs** to each player.
- Re-assign the **same ID** if a player reappears later in the match (e.g., near goal posts or after going off-frame).
- Ignore irrelevant detections (like the ball) for accurate player analysis.
- Simulate real-time detection and tracking frame-by-frame.

---

## 🛠️ How It Works

### 1. **Detection**
- The model detects players in each video frame.
- We filter out detections to **only track players**, excluding the ball.

### 2. **Tracking + Re-identification**
- Each player is assigned a unique ID the first time they are detected.
- If a player disappears temporarily and reappears later, our tracking system **recognizes and re-identifies** them using **centroid-based matching**.
- Re-appearance frames are logged for reference.

### 3. **Visualization**
- Each player is shown in the video with a **bounding box** and their assigned **Player ID**.
- Real-time output allows visual confirmation of the tracking performance.

### 4. **Summary Log**
- After the video ends, a **log file is generated** (report.txt) that clearly mentions:
  - First frame where each player appeared
  - Frames where the same player was re-identified after disappearing

---

## 🧠 Technologies Used

- **YOLOv11 (Ultralytics)** – Fine-tuned object detection model
- **OpenCV** – Video processing and display
- **Python** – Scripting and logic
- **Centroid Tracking** – For player tracking and re-ID
- **NumPy & SciPy** – For distance calculations and matrix operations

---

## 📁 Project Structure

```plaintext
📂 soccer-reid-project/
│
├── model/
│   └── best.pt                   # Trained YOLOv11 model for players & ball
│
├── video/
│   └── 15sec_input_720p.mp4      # Match video clip
│
├── track_players.py/
│   └── Code.py                   # Main tracking and detection script
│
├── report.txt                    # Auto-generated re-identification summary
