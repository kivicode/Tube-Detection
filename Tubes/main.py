'''
                                                ***INSTRUCTION***

    1) Run the code below
    2) Set up defaults for your environment
    3) WARNING! You should remember your setting because after you finish the program in debug mode, you will lose all unsaved data
    4) Enjoy

    P.S.
        Let me know in case of any questions
'''

import cv2
import numpy as np

zero_frame_file_name = 'images/b.jpg'
current_frame_file_name = 'images/a.jpg'

cv2.namedWindow('out', cv2.WINDOW_NORMAL)
cv2.resizeWindow('out', 640, 480)

''' Init debug track bars '''
# Comment/delete it after debugging
cv2.createTrackbar('threshold', 'out', 0, 255, lambda x: 0)
cv2.createTrackbar('minimal area', 'out', 0, 100000, lambda x: 0)
cv2.createTrackbar('maximum area', 'out', 0, 100000, lambda x: 0)
cv2.createTrackbar('marble threshold', 'out', 0, 1000000, lambda x: 0)

''' Settings '''
# Here are some defaults, your values may be another

debug = True  # Set to False after debugging

threshold = 48
sigma = 0  # Canny threshold

blur_scale = 6

min_area = 5000  # Filter noise (limit the smallest area of tube's contour)
max_area = 50000  # Filter noise limit the biggest area of tube's contour)

crop_width = 150  # Width of crop rectangle of image for classification
crop_height = 150  # Height of crop rectangle of image for classification

marble_threshold = 12000  # Threshold for determining the level of marbling (depends on crop sizes)


# Smart settable canny edge detection
def canny(image):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


# Find differences of zero frame and current frame and filter the noise
def crop(img):
    diff = cv2.absdiff(zero_frame, img)  # Find difference of the zero frame and current frame
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    imask = mask > threshold  # Make mask

    canvas = np.zeros_like(img, np.uint8)
    canvas[imask] = img[imask]  # Apple mask

    contours, _ = cv2.findContours(cv2.blur(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), (blur_scale,) * 2), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)  # Finding all potential tubes
    crops = []
    if not debug:
        cv2.imshow('mask', cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)  # DEBUG ONLY: draw not filtered potential tubes

    for cnt in contours:
        area = cv2.contourArea(cnt)
        try:
            if area >= min_area:
                # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)  # Visualise the contour of the tube
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
                w, h = crop_width, crop_height
                M = cv2.moments(cnt)
                x = int(M["m10"] / M["m00"]) - int(w / 2)  # Find center
                y = int(M["m01"] / M["m00"]) - int(h / 2)
                crops.append(
                    {'cropped': img[y:y + h, x:x + w], 'pos': (x, y), 'sum': np.sum(canny(img[y:y + h, x:x + w]))})
        except Exception as e:
            print(e)  # Sometimes the program can be here but don't worry it's ok
    return crops


# Sort your tubes as marbled and none marbled
def sort(images):
    marbled = []
    non_marbled = []

    for img in images:
        if img['sum'] < marble_threshold:
            marbled.append(img['pos'])
            cv2.putText(frame, 'marbled', img['pos'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            non_marbled.append(img['pos'])
            cv2.putText(frame, 'non marbled', img['pos'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return {'marbled': marbled, 'non_marble': non_marbled}


# Just for a nice code view
def main(img):
    try:
        return sort(crop(img))
    except TypeError:
        return {}


# Main loop
if __name__ == '__main__':
    while 1:
        zero_frame = cv2.imread(zero_frame_file_name)
        frame = cv2.imread(current_frame_file_name)
        if debug:
            threshold = 255 - cv2.getTrackbarPos('threshold', 'out')  # Get settings from the track bars (delete after debugging and setting up all default values)
            min_area = cv2.getTrackbarPos('minimal area', 'out')  # Get settings from the track bars (delete after debugging and setting up all default values)
            max_area = cv2.getTrackbarPos('maximum area', 'out')  # Get settings from the track bars (delete after debugging and setting up all default values)
            marble_threshold = cv2.getTrackbarPos('marble threshold', 'out')  # Get settings from the track bars (delete after debugging and setting up all default values)

        main(frame)
        cv2.imshow('out', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop the program
            break
    cv2.destroyAllWindows()
