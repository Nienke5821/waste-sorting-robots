import onnxruntime as ort
import numpy as np
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

def get_boxes_video(runtime_params, image_zed, yolo_model, coverage_threshold=0.25):
    """
    Loops through each frame grabbed from the ZED camera and runs the yolo model on this frame. Based on
    the detections of this model in this frame, the largest bounding box will retrieved and an arrow towards
    that bounding box will be drawn. In the end, each frame will be shown together with the largest bounding
    box and the directional arrow. If the largest bounding box covers a certain amount of the frame the program stops.

    Args:
        runtime_params (sl.RuntimeParameters): runtime parameters used when grabbing images.
        image_zed (sl.Mat): matrix for storing images from ZED camera.
        yolo_model (YOLO): YOLO model for the object detection.
        coverage_threshold (float): Amount of the frame area the largest bounding box should cover in 
        percentage/100. The default is set at 25% (0.25)
    """
    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.RIGHT) # LEFT or RIGHT, stereo does not work
        frame = image_zed.get_data()
        h0, w0, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        detections = yolo_model(image_rgb)[0]
        frame, largest_box, largest_box_area = draw_boxes(detections, frame)
        draw_arrow(largest_box, frame, w0, h0)

        frame_area = w0 * h0
        if largest_box_area >= coverage_threshold * frame_area:
            cv2.putText(frame, f"close to object, press any key to stop program", (10, h0-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
            cv2.imshow("YOLOv11", frame[:, :, :3])
            cv2.waitKey(0)
            break
            
        elif cv2.waitKey(1) == ord('q'):
            break

        else:
            cv2.imshow("YOLOv11", frame[:, :, :3])

def draw_boxes(detections, frame, threshold=0.2):
    """
    Adds the bounding boxes to an array if the confidence values of the objects surpass the threshold. After 
    drawing the frame, it search for the largest bounding by finding the box with the largest area. If an "largest"
    bounding box exists, this box will be drawn on the frame.

    Args:
        detections (ultralytics.engine.results.Results): output from the YOLO model containing the results.
        frame (np.ndarray): frame of the image.
        threshold (float): threshold which the confidence value of the object should surpass. Default is set to 0.2

    Returns:
        np.ndarray: frame with the largest bounding box and directional arrow drawn on, if there exist a largest 
        bounding box. Otherwise just the frame.
        tuple or none: coordinates of the largest bounding box if any, otherwise none.
        int: area of the largest bounding box.
    """
    boxes = []
    for det in detections.boxes:
        conf = det.conf.item()
        if conf < threshold:
            continue
        else:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            class_id = int(det.cls[0])
            boxes.append((conf, class_id, (x1, y1, x2, y2)))

    largest_box = None
    max_area = 0
    largest_box_info = None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    for conf, class_id, (x1, y1, x2, y2) in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = (x1, y1, x2, y2)
            largest_box_info = (conf, class_id)

    # Draw only largest box (if any)
    if largest_box is not None:
        x1, y1, x2, y2 = largest_box
        conf, class_id = largest_box_info
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_id}:{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame, largest_box, max_area

def draw_arrow(largest_box, frame, w0, h0, closeby=0.1):
    """
    Draws a directional arrow from the middle of the frame towards the middle of the largest bounding box.
    Also prints simple directions to move towards the largest bounding box. When the box is in the center, or close
    to the center, it will print center.

    Args:
        largest_box (tuple or none): coordinates of the largest bounding box if any, otherwise none.
        frame (np.ndarray): frame with the largest bounding box and directional arrow drawn on, if there
        exist a largest bounding box. Otherwise just the image frame.
        w0 (int): width of the frame.
        h0 (int): height of the frame.
        closeby (float): value that determines if the bounding box is in the center, or close to the center of the frame.
        The default is set to 0.1.
    """
    if largest_box:
        x1, y1, x2, y2 = largest_box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        frame_center = (w0 // 2, h0 // 2)

        cv2.arrowedLine(frame, frame_center, (center_x, center_y), (255, 0, 0), 3)

        dx = center_x - frame_center[0]
        dy = center_y - frame_center[1]
        
        if abs(dx) <= closeby and abs(dy) <= closeby:
            direction = "Center"
        else:
            horizontal = "Right" if dx > 0 else "Left"
            vertical = "Down" if dy > 0 else "Up"
            direction = f"Move {horizontal} {vertical}"

        cv2.putText(frame, f"Direction: {direction}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def load_video():
    """
    Open the ZED camera and initialize runtime parameters.

    Returns:
        sl.RuntimeParameters: runtime parameters used when grabbing images.
        sl.Mat: matrix for storing images from ZED camera.
    """
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA # GPU has only 6.1
    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    zed.open(init_params)
    return runtime_params, image_zed


if __name__=="__main__":

    yolo_model = YOLO("results/strategy_A/train_after_wild_litter/weights/best.pt")
    zed = sl.Camera()
    runtime_params, image_zed = load_video()
    get_boxes_video(runtime_params, image_zed, yolo_model)
    zed.close()
    cv2.destroyAllWindows()