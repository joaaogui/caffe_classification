from django.shortcuts import render

# Create your views here.

# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import cv2

get_ipython().magic(u'matplotlib inline')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from IPython.display import clear_output

import sys
import os
home_dir = os.getenv("HOME")
caffe_root = os.path.join(home_dir, 'Documents', 'caffe')
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe

if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    get_ipython().system(u'~/Documents/caffe/scripts/download_model_binary.py ~/Documents/caffe/models/bvlc_reference_caffenet')


# In[4]:

caffe.set_mode_cpu()

model_def = os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet','deploy.prototxt')
model_weights = os.path.join(caffe_root, 'models','bvlc_reference_caffenet','bvlc_reference_caffenet.caffemodel')

net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)


# In[5]:

mu = np.load(os.path.join(caffe_root, 'python','caffe','imagenet','ilsvrc_2012_mean.npy'))
mu = mu.mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

from PIL import Image
import requests
from StringIO import StringIO
import urllib


# In[6]:

net.blobs['data'].reshape(50,
                          3,
                          227, 227)


# In[7]:

def options_to_select():
    print('Choose a option')
    print('1 - Image From PC')
    print('2 - Image From Internet')
    print('3 - Image From WebCam')
    print('4 - Get out of the loop')

    return int(raw_input(''))


# In[8]:

def detect(frame):
    height, width, depth = frame.shape

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(grayscale, grayscale)

    classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    coords = []

    DOWNSCALE = 4
    minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)
    if len(faces)>0:
        for i in faces:
            x, y, w, h = [ v*DOWNSCALE for v in i ]

            coords.append((x,y,w,h))
            print x,y,w,h
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))

    print('I detected ' + str(len(faces)) + ' faces.')
    for coord in coords:
    	x,y,w,h = coord
    	print [(x,y), (x+w,y+h), (x+w, y), (x, y+h)]
    	print '\n'

    return frame


# In[9]:

def get_image_url(url):
	cap = cv2.VideoCapture(url)
	ret,img = cap.read()

	return img


# In[10]:

def get_photo_from_webcam():

	cap = cv2.VideoCapture(0)

	while(True):
	    ret, frame = cap.read()
	    img = frame.copy()

	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	    	cap.release()
	        cv2.destroyAllWindows()
	        break

	cv2.imwrite("frame.jpg", frame)
	cv2.imwrite(os.path.join(caffe_root, 'examples', 'images','frame.jpg'), frame)

	return frame


# In[11]:

def probability(image):
	transformed_image = transformer.preprocess('data', image)
	plt.imshow(image)
	plt.axis('off')

	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()
	# print output

	output_prob = output['prob'][0]

	labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')

	labels = np.loadtxt(labels_file, str, delimiter='\t')

	dogs_file = os.path.join(caffe_root, 'data','ilsvrc12','canine.txt')
	cats_file = os.path.join(caffe_root, 'data', 'ilsvrc12','feline.txt')

	labels_dogs = np.loadtxt(dogs_file, str, delimiter='\t')
	labels_cats = np.loadtxt(cats_file, str, delimiter='\t')

	top_inds = output_prob.argsort()[::-1][:]

	index = 0

	list_dogs = []
	probability_dogs = 0.0

	for i in labels:
		if i in labels_dogs:
			list_dogs.append((output_prob[index],labels[index]))
			probability_dogs += output_prob[index]
		index += 1

	index = 0

	list_cats = []
	probability_cats = 0.0

	for i in labels:
		if i in labels_cats:
			list_cats.append((output_prob[index],labels[index]))
			probability_cats += output_prob[index]
		index += 1

	list_dogs = sorted(list_dogs, reverse=True)
	list_cats = sorted(list_cats, reverse=True)

        print 'Probabilities of synsets:'

        print "\n"
        print('Canines:')
        print "\n"


	for i in list_dogs:
		dprobability , dog_class = i
		print 'Synset ' + str(dog_class)
		print 'Probability ' + str(dprobability * 100.0) + "%"

        print "\n"
        print('Felines: ')
        print "\n"

	for i in list_cats:
		cprobability , cat_class = i
		print 'Synset ' + str(cat_class)
		print 'Probability ' + str(cprobability * 100.0) + "%"

        print "\n"

	print 'Feline probability: ' + str( (probability_cats) * 100.0 ) + "%"
	print 'Canine probability: ' + str( (probability_dogs) * 100.0 ) + "%"
        print "\n"
	print 'Most probable synset:', labels[output_prob.argmax()]
	print 'Probability: ' + str(output_prob[output_prob.argmax()] * 100.0) + "%"


# In[ ]:

if __name__ == '__main__':

    while True:
        option = options_to_select()


        if option == 1:
                print '1'

                image_name = raw_input('Enter the name of the file:\n')

                image = cv2.VideoCapture(image_name)
                ret, img = image.read()

                image_recognition = caffe.io.load_image(image_name)

                probability(image_recognition)
                image_detected = detect(img)

                plt.imshow(image_detected[:,:,::-1])
                plt.title('Detection result')
                plt.axis('off')
                plt.show()

        elif option == 2:

                image_url = raw_input('Enter the image url:\n')
                image = get_image_url(image_url)

                urllib.urlretrieve(image_url, os.path.join(caffe_root, 'examples', 'images','url.jpg'))
                image_recognition = caffe.io.load_image(os.path.join(caffe_root, 'examples', 'images','url.jpg'))

                probability(image_recognition)
                image_detected = detect(image)

                plt.imshow(image_detected[:,:,::-1])
                plt.title('Detection result')
                plt.axis('off')
                plt.show()

        elif option == 3:

                frame = get_photo_from_webcam()
                image_recognition = caffe.io.load_image(os.path.join(caffe_root, 'examples', 'images','frame.jpg'))
                probability(image_recognition)
                image_detected = detect(frame)

                plt.imshow(image_detected[:,:,::-1])
                plt.title('Detection result')
                plt.axis('off')
                plt.show()


        elif option == 4:
                break
