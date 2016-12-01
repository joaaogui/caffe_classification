from django.shortcuts import render
# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}

	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		myfile = request.FILES['myfile']
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_url = fs.url(filename)
		image = _grab_image(project_root + uploaded_file_url)
		# convert the image to grayscale, load the face cascade detector,
		# and detect faces in the image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
		rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

		# construct a list of bounding boxes from the detection
		rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

		# update the data dictionary with the faces detected
		data.update({"num_faces": len(rects), "faces": rects, "success": True})

		return render(request , 'face_detector/index.html', {
			'response': data
		})

	# return a JSON response
	return render(request, 'face_detector/detect.html', {
		'response': data
	})


def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)

	# otherwise, the image does not reside on disk
	else:
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()

		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()

		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image

# Create your views here.
