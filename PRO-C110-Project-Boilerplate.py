# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf


# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)
mymodel =tf.keras.models.load_model("keras_model.h5")

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		resized_frame = cv2.resize(frame,(224,224))
		# expand the dimensions
		resized_frame = np.expand_dims(resized_frame,excess = 0)
		resized_frame = resized_frame/255
		predictions = mymodel.predict(resized_frame)
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissors = int(predictions[0][2]*100)
		print(f"rock:{rock}%,paper:{paper}%,scissors:{scissors}%,")
		# normalize it before feeding to the model
		
		# get predictions from the model
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
