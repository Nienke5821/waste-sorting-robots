import onnxruntime as ort
import numpy as np
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

def get_boxes_image(file, yolo_model):
    """
    reads the image and puts it in a frame. Based on the detections of the YOLO model on this frame, the largest 
    bounding box will retrieved and a directional arrow towards that bounding box will be drawn.

    Args:
        image_zed (sl.Mat): matrix for storing images from ZED camera.
        yolo_model (YOLO): YOLO model for the object detection.
    """
    frame = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    detections = yolo_model(image_rgb)[0]
    frame, largest_box = draw_boxes(detections, frame)
    draw_arrow(largest_box, frame)
    cv2.imshow("YOLOv11", frame[:, :, :3])

def get_boxes_video(runtime_params, image_zed, yolo_model):
    """
    Loops through each frame grabbed from the ZED camera and runs the yolo model on this frame. Based on
    the detections of this model in this frame, the largest bounding box will retrieved and an arrow towards
    that bounding box will be drawn. In the end, each frame will be shown together with the largest bounding
    box and the directional arrow.

    Args:
        runtime_params (sl.RuntimeParameters): runtime parameters used when grabbing images.
        image_zed (sl.Mat): matrix for storing images from ZED camera.
        yolo_model (YOLO): YOLO model for the object detection.
    """
    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.RIGHT) # LEFT or RIGHT
        frame = image_zed.get_data()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        detections = yolo_model(image_rgb)[0]
        frame, largest_box = draw_boxes(detections, frame)
        draw_arrow(largest_box, frame)
        cv2.imshow("YOLOv11", frame[:, :, :3])
        if cv2.waitKey(1) == ord('q'):
            break

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
    """
    boxes = []
    for det in detections.boxes:
        conf = det.conf.item()
        if conf < threshold:
            continue
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        class_id = int(det.cls[0])
        boxes.append((conf, class_id, (x1, y1, x2, y2)))

    largest_box = None
    max_area = 0
    largest_box_info = None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    for conf, class_id, (x1, y1, x2, y2) in boxes:
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # draw all the boxes
        # cv2.putText(frame, f"{class_id}:{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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

    return frame, largest_box

def draw_arrow(largest_box, frame, closeby=0.1):
    """
    Draws a directional arrow from the middle of the frame towards the middle of the largest bounding box.
    Also prints simple directions to move towards the largest bounding box. When the box is in the center, or close
    to the center, it will print center.

    Args:
        largest_box (tuple or none): coordinates of the largest bounding box if any, otherwise none.
        frame (np.ndarray): frame with the largest bounding box and directional arrow drawn on, if there
        exist a largest bounding box. Otherwise just the image frame.
        closeby (float): value that determines if the bounding box is in the center, or close to the center of the frame.
        Default is set to 0.1.
    """
    h0, w0, _ = frame.shape

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


def load_video(file):
    """
    Open the ZED camera and initialize runtime parameters.

    Returns:
        sl.RuntimeParameters: runtime parameters used when grabbing images.
        sl.Mat: matrix for storing images from ZED camera.
    """
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(file)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA # GPU has only 6.1
    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    zed.open(init_params)
    return runtime_params, image_zed


if __name__=="__main__":

    # Parameters
    yolo_model = YOLO("results/strategy_A/train_after_wild_litter/weights/best.pt")
    # file = "testdata/ZED2/HD1080_SN29184346_13-18-2qq9.svo"
    file = "testdata/ZED2/HD1080_SN29184346_13-21-43.svo"
    # file = "testdata/ZED2/Explorer_HD1080_SN29184346_13-21-12.png"

    # Check whether test data is a video or a image. Run get_boxes_video if video and 
    # get_boxe_image when it is an image.
    if file[-3:] == "svo":
        zed = sl.Camera()
        runtime_params, image_zed = load_video(file)
        get_boxes_video(runtime_params, image_zed, yolo_model)
        zed.close()
        cv2.destroyAllWindows()

    elif file[-3:] == "png":
        get_boxes_image(file, yolo_model)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No correct input for video/image is given")