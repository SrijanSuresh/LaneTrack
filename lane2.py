import cv2
import numpy as np
import glob

def calibrate_camera(calibration_images_dir, chessboard_size=(9, 6)):
    """
    Calibrate the camera using chessboard images.
    Args:
        calibration_images_dir (str): Directory containing chessboard images.
        chessboard_size (tuple): Number of inner corners per chessboard row and column.
    Returns:
        tuple: Camera matrix and distortion coefficients.
    """
    obj_points = []
    img_points = []
    
    # Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # Load calibration images
    images = glob.glob(f"{calibration_images_dir}/*.jpg")
    
    for image_path in images:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
    
    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    
    if not ret:
        raise ValueError("Camera calibration failed. Check your chessboard images.")
    
    return camera_matrix, dist_coeffs

def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Undistort an image using the camera matrix and distortion coefficients.
    """
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)

def process_frame(frame, camera_matrix, dist_coeffs):
    """
    Process a single frame for lane detection.
    """
    # Undistort the frame
    undistorted = undistort_image(frame, camera_matrix, dist_coeffs)
    
    # Convert to grayscale
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, height // 2)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    
    # Hough transform to detect lines
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=50
    )
    
    # Draw lines on the frame
    line_frame = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Combine the original frame with the line overlay
    combined = cv2.addWeighted(undistorted, 0.8, line_frame, 1, 1)
    return combined

def advanced_lane_detection(input_video, output_video, calibration_images_dir):
    """
    Perform advanced lane detection with camera calibration.
    """
    # Calibrate the camera
    print("Calibrating camera...")
    camera_matrix, dist_coeffs = calibrate_camera(calibration_images_dir)
    print("Camera calibration completed.")
    
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Initialize VideoWriter
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame for lane detection
        processed_frame = process_frame(frame, camera_matrix, dist_coeffs)
        
        # Write the processed frame to the output video
        out.write(processed_frame)
        
        # Optionally display the frame
        cv2.imshow("Lane Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "project_video.mp4"  # Path to input MP4 file
    output_video = "output2.mp4"  # Path to output MP4 file
    calibration_images_dir = "camera_cal"  # Path to chessboard images directory
    advanced_lane_detection(input_video, output_video, calibration_images_dir)
