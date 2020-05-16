from imutils import face_utils
import argparse
import dlib
import cv2
import time

detector = dlib.get_frontal_face_detector()
#cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
args = vars(ap.parse_args())

input_path=args["input"]
# load the input image, resize it, and convert it to grayscale
print("Processing file: {}".format(input_path))
if input_path[-4:] in ['.jpg', '.png', '.bmp','jpeg']:
    image=cv2.imread(input_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    print("Detections::::::::;",rects)

    # rects = dlib.rectangles()
    # rects.extend([d.rect for d in rects])
    #
    # print("Detections::::::::;", rects)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        #shape = predictor(gray, rect)
        #shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        print("rect:::::::::",rect)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        print("x y w h:::::::::;",x,y,w,h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        #for (x, y) in shape:
            #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    # show the output image with the face detections + facial landmarks
    pathOut = 'output_cnn' + input_path.split('.')[0] + ".jpg"
    cv2.imwrite(pathOut,image)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

if input_path[-4:]==".mp4":
    s=time.time()
    c=0
    cap = cv2.VideoCapture(input_path)
    pathOut = 'output_2'+input_path.split('.')[0]+".mp4"
    width=int(cap.get(3))
    height=int(cap.get(4))
    fps=float(cap.get(5))
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'MPEG'), fps, (1920,1080))

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, image = cap.read()
        if ret == True:
            # Our operations on the frame come here
            dim = (1920,1080)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 1)
            if len(rects) != 0:
                c+=1
            print("Detections::::::::;", rects)

            # loop over the face detections
            for (i, rect) in enumerate(rects):
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

    print("Total number of detection:",c)
    print("Total time for execution:",time.time()-s)
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

