import cv2
from ultralytics import YOLO, SAM
import numpy as np
from math import atan2,degrees


def get_houghvalues(points, width, height):
    """Find the rho and theta values for the hough transform that represent
    the line that goes through the most points

    Args:
        points (list): The points through which we want to draw a straight line
        width (int): the width of the image
        height (int): the height of the image
    
    Returns:
        Tuple[float, float]: the theta and rho value that define the best line
    """
    blankimage = np.zeros((width, height), dtype=np.uint8) # create blank image to find hough line
    best_rho = None
    best_theta = None

    # Plot the points on the image
    for point in points:
        cv2.circle(blankimage, point, radius=1, color=255, thickness=-1) # add points to image

    # Use Hough Transform to detect lines
    lines = cv2.HoughLines(blankimage, rho=1, theta=np.pi / 180, threshold=1) # calculate lines

    max_votes = 0
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # number of points that are (almost) on the line
            votes = np.sum(np.isclose(lines[:, 0, 0], rho) & np.isclose(lines[:, 0, 1], theta))
            if votes > max_votes: # find line with highest count
                max_votes = votes
                best_rho = rho
                best_theta = theta
    return best_rho, best_theta


def get_center_line(rho, theta, centered):
    """Using the rho and theta value from Hough transform, compute the startpoint
    and endpoint of the line that goes through part of the points

    Args:
        rho (float): the rho value of the best hough transform line
        theta (float): the theta value of the best hough transform line
        centered (list): the points through which the line should go

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int], float]: the startpoint and endpoint
        of the line, together with the degree that the line is rotated compared to the 
        y-axis.
    """
    sorted_points = np.array(sorted(centered, key=lambda pt: pt[1]))
    max_point = []
    min_point = []
    for point in sorted_points:
        x = point[0]
        actual_y = point[1]
        # use formula to find the supposed y-value if the [x,y] point is on the line
        calc_y = (-np.cos(theta) / np.sin(theta)) * x + (rho / np.sin(theta))
        if np.abs(actual_y - calc_y) <= 10: # points does not have to be exactly on the line
            # on line
            if len(max_point) == 0 or max_point[1] < actual_y:
                max_point = point
            if len(min_point) == 0 or min_point[1] > actual_y:
                min_point = point
    if len(max_point) > 0:               
        xdiff = max_point[0] - min_point[0]
        ydiff = max_point[1] - min_point[1]
        angled = 1 * (degrees(atan2(ydiff, xdiff)) - 90) # calculate rotation of the line
        return (min_point[0],  min_point[1]) , (max_point[0], max_point[1]), angled
    return [], [], None # no line


def draw_image_detected(image, points, startpca, endpca, center_x, x_movement, y_movement, angle):
    """Draw the outline of the bottle together with the line that we want to center.
    Also display the movement instructions

    Args:
        image (numpy.ndarray): The image received by the camera
        points (list): The points that represent the outline of the object
        startpca (tuple): The startpoint of the line we want to center
        endpca (tuple): The endpoint of the line we want to center
        center_x (int): the x-coordinate that is the center of the image
        x_movement (string): the movement instruction for the x-axis
        y_movement (string): the movement instruction for the y-axis
        angle (string): the instruction for the rotation
    """
    for point in points: # previous detected outline
        cv2.circle(image, point, radius=1, color=(255, 0, 0), thickness=-1)
    cv2.line(image, startpca, endpca, (0, 255, 0), 2) # line through previous detected center
    # give instructions
    cv2.putText(image, x_movement, [10, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if y_movement is not None:
        cv2.putText(image, y_movement, [center_x - 40, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if angle is not None:
        cv2.putText(image, str(angle), [center_x + 60, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def draw_image_not_detected(image, center_x, previous_x_direction = None, previous_y_direction = None, previous_angle = None):
    """Draw the previous instructions from when the bottle was detected on the screen.

    Args:
        image (numpy.ndarray): The image received as output from the camera
        center_x (int): The center x-coordinate of the image
        previous_x_direction (string, optional): The previous movement instruction along the x-axis. Defaults to None.
        previous_y_direction (string, optional): The previous movement instruction along the y-axis. Defaults to None.
        previous_angle (str, optional): The previous rotation instruction. Defaults to None.
    """
    if previous_angle is None and previous_x_direction is None: # never detected before
        cv2.putText(image, "Not detected", [10, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif previous_x_direction != "done": # move left or right
        cv2.putText(image, previous_x_direction, [10, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif previous_y_direction != "done": # move up or down
        cv2.putText(image, previous_y_direction, [center_x - 40, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif previous_angle == "done": # finished
        cv2.putText(image, previous_x_direction, [10, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, previous_angle, [center_x, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else: #rotate either clockwise or counterclockwise
        cv2.putText(image, previous_x_direction, [10, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if previous_angle < 0:
            # only one degree since bottle might already have been turned correctly
            cv2.putText(image, "-1", [center_x, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(image, "1", [center_x, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)



def draw_lines(image, center_x, center_y, width, height):
    """Draw the center lines between which the line has to be on the x-axis and y-axis, and the rotation
    between which the line should be.

    Args:
        image (numpy.ndarray): The image received as output from the camera
        center_x (int): The center of the image on the x-axis
        center_y (int): The center of the image on the y-axis
        width (int): The width of the image
        height (int): The height of the image
    """
    # straight lines
    cv2.line(image, (center_x - slack, 0), (center_x - slack, height), (0,0,255), 2)
    cv2.line(image, (center_x + slack, 0), (center_x + slack, height), (0,0,255), 2)
    cv2.line(image, (0, center_y - slack), (width, center_y - slack), (0,0,255), 2)
    cv2.line(image, (0, center_y + slack), (width, center_y + slack), (0,0,255), 2)

    # rotation allowed is 5 degrees
    angle = 5  # degrees
    angle_rad = np.deg2rad(angle)
    angle2 = -5
    angle_rad2 = np.deg2rad(angle2)
    # calculate x-offset based on the height of the image and the angle
    x_offset = int(np.tan(angle_rad) * height)
    x_offset2 = int(np.tan(angle_rad2) * height)
    # start and end points for the line
    start_point = (center_x - x_offset // 2, 0)  
    end_point = (center_x + x_offset // 2, height) 
    start_point2 = (center_x - x_offset2 // 2, 0) 
    end_point2 = (center_x + x_offset2 // 2, height)  
    # Draw the lines
    cv2.line(image, start_point, end_point, (0, 255, 0), 3)
    cv2.line(image, start_point2, end_point2, (0, 255, 0), 3)

    
def get_hori_movement(x_loc, center_x):
    """Get the direction in which the claw has to move along the x-axis

    Args:
        x_loc (int): The location of the center of the line on the x-axis
        center_x (int): The center of the image on the x-axis

    Returns:
        string: the direction in which the claw has to move
    """
    if x_loc < center_x - slack:
        return "left"
    elif x_loc > center_x + slack:
        return "right"
    else:
        return "done"
    
def get_verti_movement(y_loc, center_y):
    """Get the direction in whcih the claw has to move along the y-axis

    Args:
        y_loc (int): The location of the center of the line along the y-axis
        center_y (int): The center of the image on the y-axis

    Returns:
        string: The direction in which the claw has to move along the y-axis.
    """
    if y_loc < center_y - slack:
        return "down"
    elif y_loc > center_y + slack:
        return "up"
    else:
        return "done"
    

def get_points_sam(results):
    """Retrieve all points of the mask

    Args:
        results (list): the output from the SAM2 or YOLO model

    Returns:
        list: the x and y coordinates of all points of the mask
    """
    points = []
    result = results[0]
    if result.masks is not None:
        for mask in result.masks.data:
            y, x = np.where(mask.cpu().numpy() > 0)  # convert tensor to numpy
            points.extend(zip(x, y))
    return points


def get_box(image):
    """Get the bounding box of the bottle in the image

    Args:
        image (numpy.ndarray): The image that is the output of the camera

    Returns:
        list: The coordinates of the bounding box surrounding a detected bottle
    """
    results = detectBottle(image)
    result = results[0]
    for result in results:
        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                return [x1, y1, x2, y2]
    return []

def get_avg_points_over_x(points):
    """For each x-value, take the average of all y-values and return all the average points
    Args:
        points (list): the points containing the x and y coordinates that describe the mask

    Returns:
        list: a list of points where each y-value is the average y-value of the corresponding x-value
    """
    sorted_points = np.array(sorted(points, key=lambda pt: pt[0])) # sort based on x-value
    prev_x = 0
    middle = []
    subpoints = []
    for item in sorted_points:
        x = item[0]
        if prev_x == x: # same x-value
            subpoints.append(item[1])
        else:
            if len(subpoints) > 0:
                middle.append([x, int(np.mean(subpoints))]) # add x-value and average y-value
            subpoints = [item[1]] # remove old y-values and add new one
        prev_x = x
    middle.append([x, int(np.mean(subpoints))]) # also add last point
    return middle

def get_avg_points_over_y(points):
    """For each y-value, take the average of all x-values and return all the average points
    Args:
        points (list): the points containing the x and y coordinates that describe the mask

    Returns:
        list: a list of points where each x-value is the average x-value of the corresponding y-value
    """
    sorted_points = np.array(sorted(points, key=lambda pt: pt[1])) # sort based on y-value
    prev_y = 0
    middle = []
    subpoints = []
    for item in sorted_points:
        y = item[1]
        if prev_y == y: # same y-value
            subpoints.append(item[0]) # add x-value
        else:
            if len(subpoints) > 0:
                middle.append([int(np.mean(subpoints)), y]) # add average of x-values and y-value
            subpoints = [item[0]] # remove old x-values and replace with new one
        prev_y = y # update y
    middle.append([int(np.mean(subpoints)), y]) # also add the last point
    return middle                
    
def get_outline_sam(results):
    """Get the x- and y-coordinates of the outline of the bottle to show what it detects on the screen

    Args:
        results (list): the output from the SAM2 model

    Returns:
        numpy.ndarray: the x and y coordinates of all points that are part of the outline
    """
    result = results[0]
    if result.masks is not None:
        for mask in result.masks.xy:
            # Convert mask to int32
            points = np.array(mask, dtype=np.int32)
        return points
    else: # no object detected
        return []  
    
detectBottle = YOLO('best.pt').to('cuda') # load the model from the Detection part  
# initialize values for camera 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)
# initialize variables
slack = 20
angle_slack = 5
previous_angle = None
previous_x_direction = None
previous_y_direction = None
model_seg = SAM("sam2.1_t.pt").to('cuda') # load segmentation model
while True:
    success, image = cap.read() # get the image
    height, width, _ = image.shape
    center_x = int(width / 2) # center of the image
    center_y = int(height / 2)
    bbox = get_box(image) # bounding box of bottle
    mask_points = []
    if len(bbox) > 0: # there is a bottle detected
        results = model_seg(image, bboxes = bbox) # apply SAM2
        mask_points = get_points_sam(results)
    rho = None
    draw_lines(image, center_x, center_y, width, height)
    start_avg = []
    if len(mask_points) > 0: # there is a bottle detected
        if bbox[2] - bbox[0] >= bbox[3] - bbox[1]: # width >= height:
            avg_points = get_avg_points_over_x(mask_points)
        else: # height > width
            avg_points = get_avg_points_over_y(mask_points)
        rho, theta = get_houghvalues(avg_points, width, height)
    if rho != None: # there is a line
        start_avg, end_avg, angle = get_center_line(rho, theta, avg_points)    
    if len(start_avg) > 0: # there is a line
        # get instructions
        center_bottle_x = int(np.abs(start_avg[0] - end_avg[0]) / 2 + min(start_avg[0], end_avg[0]))
        x_movement = get_hori_movement(center_bottle_x, center_x)
        center_bottle_y = int(np.abs(start_avg[1] - end_avg[1]) / 2 + min(start_avg[1], end_avg[1]))
        y_movement = get_verti_movement(center_bottle_y, center_y)
        previous_x_direction = x_movement
        previous_angle = angle

        # draw image
        outline_points = get_outline_sam(results)
        if x_movement != "done": # not center x positioned
            draw_image_detected(image, outline_points, start_avg, end_avg, center_x, x_movement, None, None)
        elif y_movement != "done": # not center y positioned
            draw_image_detected(image, outline_points, start_avg, end_avg, center_x, "done", y_movement, None)
        elif abs(angle) >= angle_slack:  # not oriented upright
            draw_image_detected(image, outline_points, start_avg, end_avg, center_x, "done", "done", angle)
        else: # positioned correctly
            draw_image_detected(image, outline_points, start_avg, end_avg, center_x, "done", "done", "done")

    else: # not detected
        if previous_angle is None and previous_x_direction is None and previous_y_direction is None: # never detected before
            draw_image_not_detected(image, center_x)
        elif previous_x_direction != "done": # move left or right
            draw_image_not_detected(image, center_x, previous_x_direction)
        elif previous_y_direction != "done": # move up or down
            draw_image_not_detected(image, center_x, "done", previous_y_direction)
        elif abs(previous_angle) < angle_slack: # positioned correctly
            draw_image_not_detected(image, center_x, "done", "done", "done")
        else: #rotate either clockwise or counterclockwise
            draw_image_not_detected(image, center_x, "done", "done", previous_angle)
    cv2.imshow("instructions", image) # show image
    if cv2.waitKey(1) == ord('q'):
        # end the loop if user presses "q"
        break