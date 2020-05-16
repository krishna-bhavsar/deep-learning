from imutils import face_utils
import argparse
import dlib
import cv2
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
args = vars(ap.parse_args())

input_path=args["input"]
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

s = time.time()
c = 0

print("Processing file: {}".format(input_path))
# img=cv2.imread(input_path)
# # The 1 in the second argument indicates that we should upsample the image
# # 1 time.  This will make everything bigger and allow us to detect more
# # faces.
# dets = cnn_face_detector(img, 1)
# '''
# This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
# These objects can be accessed by simply iterating over the mmod_rectangles object
# The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

# It is also possible to pass a list of images to the detector.
#     - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

# In this case it will return a mmod_rectangless object.
# This object behaves just like a list of lists and can be iterated over.
# '''
# print("Number of faces detected: {}".format(len(dets)))
# if len(dets) != 0:
#     c += 1
# print("Detections::::::::;", dets)
# rects = dlib.rectangles()
# rects.extend([d.rect for d in dets])

# for (i, rect) in enumerate(rects):
#     # determine the facial landmarks for the face region, then
#     # convert the facial landmark (x, y)-coordinates to a NumPy
#     # array
#     # shape = predictor(gray, rect)
#     # shape = face_utils.shape_to_np(shape)
#     # convert dlib's rectangle to a OpenCV-style bounding box
#     # [i.e., (x, y, w, h)], then draw the face bounding box
#     (x, y, w, h) = face_utils.rect_to_bb(rect)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# cv2.imshow('Output', img)
# cv2.imwrite('output_cnn' + input_path.split('.')[0] + ".jpg",img)
# print("Total number of detection:", c)
# print("Total time for execution:", time.time() - s)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
cap = cv2.VideoCapture(input_path)
pathOut = 'output_cnn' + input_path.split('.')[0] + ".mp4"
# width = int(cap.get(3))
# height = int(cap.get(4))
fps = float(cap.get(5))
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'MPEG'), fps, (1920, 1080))

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, image = cap.read()
    if ret == True:
        # Our operations on the frame come here
        dim = (1920, 1080)

        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        dects = cnn_face_detector (rgb_image, 1)
        if len(dects) != 0:
            c += 1
        print("Detections::::::::;", dects)
        # loop over the face detections

        rects = dlib.rectangles()
        rects.extend([d.rect for d in dects])

        print("Detections::::::::;", rects)

        for (i, rect) in enumerate(rects):
            print("in for loop")
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            # shape = predictor(gray, rect)
            # shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame
        out.write(image)
        cv2.imshow('Output', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print("Total number of detection:", c)
print("Total time for execution:", time.time() - s)
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()