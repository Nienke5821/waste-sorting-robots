from ultralytics import SAM, YOLO
import cv2
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
    
def get_mask_points(results):
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
            y, x = np.where(mask.cpu().numpy() > 0)  # Convert tensor to numpy
            points.extend(zip(x, y))
        return points
    else: # There is no object detected
        return []  
     
def get_outline_points(results):
    """Retrieve all points that are part of the outline of the object

    Args:
        results (list): the output from the SAM2 or YOLO model

    Returns:
        numpy.ndarray: the x and y coordinates of all points that are part of the outline
    """
    result = results[0]
    if result.masks is not None:
        for mask in result.masks.xy:
            # Convert type mask to int32
            points = np.array(mask, dtype=np.int32)
        return points
    else: # There is no object detected
        return []  
     
def get_avg_points(points):
    """For each x-value, take the average of all y-values and return all the average points

    Args:
        points (list): the points containing the x and y coordinates that describe the mask

    Returns:
        list: a list of points where each y-value is the average y-value of the corresponding x-value
    """
    sorted_points = np.array(sorted(points, key=lambda pt: pt[0])) # Sort list based on x-value
    prev_x = 0
    middle = []
    subpoints = []
    for item in sorted_points:
        x = item[0]
        if prev_x == x: # same x-value
            subpoints.append(item[1])
        else: # new x-value
            if len(subpoints) > 0:
                middle.append([x, int(np.mean(subpoints))]) # take the average of all y-values
            subpoints = [item[1]] # remove all y-values from the previous x-value and add the new one
        prev_x = x
    middle.append([x, int(np.mean(subpoints))]) # add the last point
    return middle

def get_min_max_center_points(points):
    """for each x-value, compute the average of the lowest and highest corresponding y-value. Return all these points.

    Args:
        points (numpy.ndarray): The x and y coordinates of all points that are part of the outline of the object

    Returns:
        numpy.ndarray: list of points that represent the x-coordinate and corresponding average y-coordinate
    """
    sorted_points = np.array(sorted(points, key=lambda pt: pt[0])) # sort list based on x-values
    prev_x = 0
    middle = []
    min_y = 0
    max_y = 0
    for item in sorted_points:
        x = item[0]
        if prev_x == x:
            # the same
            if min_y > item[1]:
                min_y = item[1]
            elif max_y < item[1]:
                max_y = item[1]
        else:
            # start over
            middle_y = (min_y + max_y) / 2 # take average
            if min_y != max_y: # to filter out wrong values
                middle.append([x, int(middle_y)])
            min_y = item[1]
            max_y = item[1]
        prev_x = x
    middle.append([x, int((min_y + max_y) / 2)]) # add average point
    return np.array(middle)

def get_houghline(points, width, height):
    """Compute a straight line that goes through part of the points. Hough transform is used to find the 
    line trough which most points go. From this point the startpoint and endpoint are computed

    Args:
        points (list): The points through which we want to draw a straight line
        width (int): the width of the image
        height (int): the height of the image

    Returns:
        Tuple[int, int, int, int]: the x and y coordinates that represent the startpoint end endpoint of the line
    """
    blankimage = np.zeros((width, height), dtype=np.uint8) # create blank image to find hough line
    for point in points:
        cv2.circle(blankimage, point, radius=1, color=255, thickness=-1) # add points to the image
    lines = cv2.HoughLines(blankimage, rho=1, theta=np.pi / 180, threshold=10) # calculate the lines
    

    best_line = None
    max_votes = 0
    for line in lines:
        rho, theta = line[0]
        # number of points that are (almost) on the line
        votes = np.sum(np.isclose(lines[:, 0, 0], rho) & np.isclose(lines[:, 0, 1], theta))
        if votes > max_votes: # find line with highest count
            max_votes = votes
            best_line = (rho, theta)

    # Draw the line with the most points
    if best_line:
        rho, theta = best_line

    sorted_points = np.array(sorted(points, key=lambda pt: pt[1])) # sort points based on y-value
    max_point = []
    min_point = []
    for point in sorted_points:
        x = point[0]
        actual_y = point[1]
        # use formula to find the supposed y-value if the [x, y] point is on the line
        calc_y = (-np.cos(theta) / np.sin(theta)) * x + (rho / np.sin(theta))
        if np.abs(actual_y - calc_y) <= 10: # point can be very close to the line
            # on line
            if len(max_point) == 0 or max_point[1] < actual_y:
                max_point = point
            if len(min_point) == 0 or min_point[1] > actual_y:
                min_point = point
    return min_point[0],  min_point[1], max_point[0], max_point[1]

def get_pca_line(points):
    """Compute the line that represents a straight line through the points calcuated using the first principal component
    calculated by the Principle Component Analysis method

    Args:
        points (list): The points through which the line is supposed to go

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: the startpoint end endpoint of the line represented by x and y coordinates
    """
    pca = PCA(n_components=2)
    pca.fit(points)
    center = pca.mean_ # The center of the points which is also the center of the line
    direction = pca.components_[0] # biggest variance
    scale = 200  # make the line long enough
    startpca = (int(center[0] - scale * direction[0]), int(center[1] - scale * direction[1])) # compute startpoint of the line
    endpca = (int(center[0] + scale * direction[0]), int(center[1] + scale * direction[1])) # compute endpoint of the line
    return startpca, endpca

# draw and save all iamges
x1, y1, x2, y2 = np.array([28, 69, 495, 269]) # the bounding box of the bottle in "crumpledbottle.jpg"
model_seg = SAM("sam2.1_t.pt")
results = model_seg("crumpled-bottle.jpg", bboxes = [x1, y1, x2, y2]) # Using the SAM2 model

image = cv2.imread('crumpled-bottle.jpg') # load the image
height, width, _ = image.shape
mask_points = get_mask_points(results) # get the mask
for point in mask_points: # draw the mask
        cv2.circle(image, point, radius=1, color=(128, 0, 128), thickness=-1)

avg_points = get_avg_points(mask_points) # get the average points from the mask
for point in avg_points: # draw the average points
        cv2.circle(image, point, radius=1, color=(0, 0, 0), thickness=-1)
cv2.imwrite("SAM2 avg points.jpg", image) 
x1h, y1h, x2h, y2h = get_houghline(avg_points, width, height) # compute a straight line through part of the points
cv2.line(image, (x1h, y1h), (x2h, y2h), (0,165,255), 2) # draw the line
cv2.imwrite("SAM2 hough.jpg", image) 

image = cv2.imread('crumpled-bottle.jpg') # remove everything drawn on the image
for point in mask_points: # draw the mask
        cv2.circle(image, point, radius=1, color=(128, 0, 128), thickness=-1)
startpca, endpca = get_pca_line(mask_points) # compute the PCA line
cv2.line(image, startpca, endpca, (0, 165, 255), 2) # draw the line
cv2.imwrite("SAM2 pca mask.jpg", image) 
image = cv2.imread('crumpled-bottle.jpg') # remove everything drawn on the image
outline_points = get_outline_points(results) # get the outline of the bottle

for point in outline_points:
        cv2.circle(image, point, radius=1, color=(128, 0, 128), thickness=-1)
min_max_center_points = get_min_max_center_points(outline_points) # get the center points
for point in min_max_center_points:
        cv2.circle(image, point, radius=1, color=(0, 0, 0), thickness=-1)
cv2.imwrite("SAM2 min max center points.jpg", image) 

image = cv2.imread("crumpled-bottle.jpg")
segmentBottle = YOLO("yolov8s-seg.pt") # Switch to the YOLO segmentation model

img = Image.open("crumpled-bottle.jpg")
results = segmentBottle(img)
yolo_outline_points = get_outline_points(results)
for point in yolo_outline_points:
        cv2.circle(image, point, radius=1, color=(128, 0, 128), thickness=-1)

min_max_center_points_yolo = get_min_max_center_points(yolo_outline_points)
for point in min_max_center_points_yolo:
        cv2.circle(image, point, radius=1, color=(0, 0, 0), thickness=-1)
cv2.imwrite("YOLO min max center points.jpg", image)


# print number of points used in the report
print("yolo outline count: ", len(yolo_outline_points))
print("sam2 outline points: ", len(outline_points))

print("yolo outline avg points: ", len(min_max_center_points_yolo))
print("SAM2 outline avg points: ", len(min_max_center_points))
print("SAM2 mask avg points: ", len(avg_points))