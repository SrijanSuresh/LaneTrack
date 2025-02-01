# LaneTrack

**LaneTrack** is a real-time lane detection system that processes road video footage to accurately identify and overlay lane boundaries. The system leverages traditional computer vision techniques and is designed for efficiency, scalability, and visual clarity.

---

## 1. Problem Statement and Goals

- **Objective:** Develop a lane detection system capable of accurately identifying lane boundaries in video frames in real-time.
- **Key Goals:**
  - Detect and visualize lane markings in road videos.
  - Operate under varying road conditions, including curves and intersections.
  - Overlay detected lanes on the original video for clear visual feedback.

---

## 2. Scope of the Project

- **Inclusions:**
  - Lane detection via image preprocessing (grayscale, Gaussian blur, and Canny edge detection).
  - Region-of-interest (ROI) masking to focus on areas with lane markings.
  - Lane detection using the Hough Transform and linear regression.
  - Real-time video processing (30 FPS).
- **Exclusions:**
  - Vehicle detection and advanced road sign recognition, which are considered for future work.

---

## 3. System Architecture and Methodology

### Pipeline Overview:
1. **Frame Acquisition:** Extract video frames from input files.
2. **Preprocessing:** Convert frames to grayscale and apply Gaussian blur to reduce noise.
3. **Edge Detection:** Apply the Canny edge detector.
4. **ROI Masking:** Isolate the region most likely to contain lane markings.
5. **Line Detection:** Use the Hough Transform to detect lane lines.
6. **Line Averaging:** Smooth the detected lines using linear regression.
7. **Result Overlay:** Draw the lane lines in a distinct color on the original frame.

### Tools & Libraries:
- **OpenCV:** Image processing and computer vision functions.
- **NumPy:** Array and numerical operations.
- **MoviePy:** Video frame extraction and video writing.
- **Matplotlib:** Visual debugging and plotting.

---


## 4. Results and Analysis

- **Accuracy:** Achieved approximately 85% accuracy under typical road conditions.
- **Performance:** Processed video frames at 30 FPS in real-time.
- **Challenges:** 
  - Faded or poorly visible lane markings.
  - Varying lighting conditions and extreme road curvatures.
  
---

## 5. Future Work

- **Deep Learning Integration:** Explore CNN-based models (e.g., LaneNet) for improved robustness.
- **Vehicle Detection:** Incorporate models like YOLO or Faster R-CNN to detect and track vehicles.
- **Multi-Sensor Fusion:** Combine data from cameras, LIDAR, and radar for enhanced detection accuracy.

---

## 6. Credits and Data Sources

- **Datasets:** Udacity Self-Driving Car Dataset, Comma.ai, and publicly available YouTube footage.
- **Calibration Images:** Chessboard images from the OpenCV camera calibration tutorial.
- **Libraries:** OpenCV, NumPy, MoviePy, Matplotlib.
