#!/usr/bin/env python3

import rospy
from tutorial_controller.msg import ImpedanceState
from geometry_msgs.msg import Quaternion
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Pose
import cv2
from ultralytics import YOLO, SAM
import numpy as np
import functools
import tf.transformations as tft
from math import atan2,degrees
from scipy.spatial.transform import Rotation as R


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
    
# This function is called every time a message about the robot state is received
def callbackState(msg):
    """
    Function is called every time a message about the state of the robot is received. It saves the initial pose,
    current pose and transformation matrix of the robot to the camera. A pose message from the camera to the robot is created 
    for the robot to camera transformation matrix. 

    Args:
        msg (FrankaState): ROS message type about the state of the robot.
    """

    global initial_pose, current_pose, T_cam_to_robot
    T_cam_to_robot = np.array(msg.O_T_EE).reshape((4, 4)).transpose()
    
    poseMsg_cam_to_robot = ImpedanceState() # convert it to a pose message as required by the low-level controller

    poseMsg_cam_to_robot.pose.position.x = T_cam_to_robot[0,3]
    poseMsg_cam_to_robot.pose.position.y = T_cam_to_robot[1,3]
    poseMsg_cam_to_robot.pose.position.z = T_cam_to_robot[2,3]
    quaternion_cam_to_robot = tft.quaternion_from_matrix(T_cam_to_robot)
    poseMsg_cam_to_robot.pose.orientation = Quaternion(*quaternion_cam_to_robot)

    initial_pose = poseMsg_cam_to_robot
    current_pose = poseMsg_cam_to_robot

def compute_depth(bbox, axis, fx=142, fy=164, real_height=0.22, parts_detect=2/3): 
    """
    Computes the depth of the detected bottle using pinhole method. When the width bounding box > height bounding box, focal length x (fx) and the width will be used to compute the depth.
    When the width bounding box < height bounding box, focal length y (fy) and the height will be used to compute the depth. However since most of the time, when the bottle lays
    horizontally (width > height), only parts of the bottle gets detected most of the time. the deth can recomputed using real height * parts_detect. Parts_detect is 2/3 at default as this is the fraction of the parts of th bottle that the YOLO model detecting for horizontal bottle (most of the time).

    Args:
        bbox (list): the coordinates of the bounding box surrounding a detected bottle
        axis (int): 0 if the width of the bounding box of the detected bottle is greater than the height, otherwise 1. Since the focal length of the x and y can be different, these are calculated seperately.  
        fx (int): focal length of the x-axis. Default is set to 142, as it has been calculated that from a +-30cm distance, a bottle with height +-22cm, the focal length is ~142 (bbox was +-119). This is calculated using the pinhole camera model.
        fy (int): focal length of the y-axis. Default is set to 164, as it has been calculated that from a +-30cm distance, a bottle with height +-22cm, the focal length is ~164 (bbox was +-104). This is calculated using the pinhole camera model. However, the detection algorithm did not detect the whole bottle. 
        real_height: real height of a plastic bottle. For calculations a 500ml plastic bottle is used with height 22cm.

    Returns:
        float: depth of the bottle detected.
    """
    if axis == 0:
        width = bbox[2] - bbox[0]
        adjusted_width = real_height * parts_detect
        z = (fx * adjusted_width) / width

    else:
        height = bbox[3] - bbox[1]
        z = (fy * real_height) / height
    return z

def pixel_to_real(x_pix, y_pix, z, width, height, fx=142, fy=164):
    """
    Converts the pixel values to real world values using the focal length calculated by using the reference frame.

    Args:
        x_pix (int): x position in pixels of the detected bottle.
        y_pix (int): y posiition in pixels of the detected bottle.
        z (float): z position in the real world of the detected bottle.
        width (int): the width of the image.
        height (int): the height of the image.
        fx (int): focal length of the x-axis. Default is set to 142, as it has been calculated that from a +-30cm distance, a bottle with height +-22cm, the focal length is ~142 (bbox was +-119). This is calculated using the pinhole camera model.
        fy (int): focal length of the y-axis. Default is set to 164, as it has been calculated that from a +-30cm distance, a bottle with height +-22cm, the focal length is ~164 (bbox was +-104). This is calculated using the pinhole camera model. However, the detection algorithm did not detect the whole bottle. 

    Returns:
        x (float): x position in the real world of the detected bottle.
        y (float): y posiition in the real world of the detected bottle.
    """
    hor_center = width // 2
    ver_center = height // 2
    x = (x_pix - hor_center) * z / fx
    y = -(y_pix - ver_center) * z / fy # increase upwards

    return x, y

def perpendicular_rotation(theta):
    """
    Computes the rotation of the claw, which is perpendicular to the hough transform line.

    Args:
        theta (float): the theta value of the best hough transform line.

    Returns:
        numpy.ndarray: array with the orientation of x, y, z and w (quaternion).
    """
    orientation = R.from_euler('z', theta) # theta is already perpendicular from the hough line
    quaternion = orientation.as_quat() 
    return quaternion


if __name__=="__main__":

    # initialize ROS and publisher
    rospy.init_node("target_pose_publisher")
    pub = rospy.Publisher("/panda_1/target_pose", ImpedanceState, queue_size=10)

    # Subscribe to robot state
    rospy.Subscriber('/panda_1/franka_state_controller/franka_states', FrankaState, functools.partial(callbackState))
 
    detectBottle = YOLO('best.pt') # load detection model

    # initialize values for camera 
    cap = cv2.VideoCapture(1) # 1 for extern camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    cap.set(cv2.CAP_PROP_FPS, 15)

    # initialize variables
    previous_target_pose = None
    lengthX_to_gripper = 0 # default set to 0,  x distance from gripper to camera in meters
    lengthY_to_gripper = 0 # default set to 0,  y distance from gripper to camera in meters

    rate = rospy.Rate(10)

    model_seg = SAM('sam2.1_t.pt') # load segmentation model

    while not rospy.is_shutdown():
        success, image = cap.read() # get the image

        if not success:
            rospy.logwarn("Failed to read from camera.")
            continue

        height, width, _ = image.shape
        center_x = int(width / 2) # center of the image
        center_y = int(height / 2)
        bbox = get_box(image) # bounding box of bottle
        mask_points = []
        if len(bbox) > 0: # there is a bottle detected
            results = model_seg(image, bboxes = bbox) # apply SAM2
            mask_points = get_points_sam(results)
        rho = None
        start_avg = []
        if len(mask_points) > 0: # there is a bottle detected
            if bbox[2] - bbox[0] >= bbox[3] - bbox[1]: # width >= height:
                avg_points = get_avg_points_over_x(mask_points)
                target_z = compute_depth(bbox, 0)
            else: # height > width
                avg_points = get_avg_points_over_y(mask_points)
                target_z = compute_depth(bbox, 1)
            rho, theta = get_houghvalues(avg_points, width, height)
        if rho != None: # there is a line
            start_avg, end_avg, angle = get_center_line(rho, theta, avg_points)    
        if len(start_avg) > 0: # there is a line
            # get instructions
            center_bottle_x = int(np.abs(start_avg[0] - end_avg[0]) / 2 + min(start_avg[0], end_avg[0]))
            center_bottle_y = int(np.abs(start_avg[1] - end_avg[1]) / 2 + min(start_avg[1], end_avg[1]))

            target_x = center_bottle_x
            target_y = center_bottle_y
            target_x, target_y = pixel_to_real(target_x, target_y, target_z, width, height)
            target_x = target_x - lengthX_to_gripper
            target_y = target_y - lengthY_to_gripper

            quaternion_bottle_to_cam = perpendicular_rotation(angle)
            p_bottle_to_cam = np.array([target_x, target_y, target_z]) # translation matrix bottle to camera

            rotation_bottle_to_cam = R.from_quat(quaternion_bottle_to_cam).as_matrix()
            T_bottle_to_cam = np.eye(4)
            T_bottle_to_cam[:3,:3] = rotation_bottle_to_cam
            T_bottle_to_cam[3, 3] = p_bottle_to_cam

            T_bottle_to_robot = T_cam_to_robot @ T_bottle_to_cam # compute bottle to robot target pose by doing a coordinate frame transformation (pose from bottle to cam into cam to robot frame)

            poseMsg_bottle_to_robot = ImpedanceState() # create pose Message for the pose of the bottle to the robot as needed for the low-level controller
            poseMsg_bottle_to_robot.pose.position.x = T_bottle_to_robot[0,3]
            poseMsg_bottle_to_robot.pose.position.y = T_bottle_to_robot[1,3]
            poseMsg_bottle_to_robot.pose.position.z = T_bottle_to_robot[2,3]
            quaternion_bottle_to_robot = tft.quaternion_from_matrix(T_bottle_to_robot)
            poseMsg_bottle_to_robot.pose.orientation = Quaternion(*quaternion_bottle_to_robot)

            pub.publish(poseMsg_bottle_to_robot) # publish target pose
            previous_target_pose = poseMsg_bottle_to_robot


        elif previous_target_pose is not None:
            pub.publish(previous_target_pose)
            rospy.logwarn("Bottle not detected. Using previous target.")

        else:
            rospy.loginfo("No bottle detected, and no previous target available.")


        cv2.imshow("Camera:", image) # show camera view with bounding boxes
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()
