import numpy as np
import cv2
import time
from matplotlib import pyplot as plt 

        
#Function that Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image



path_model = "models/"

# Read Network
model_name = "model-f6b98070.onnx"; # MiDaS v2.1 Large
#model_name = "model-small.onnx"; # MiDaS v2.1 Small


# Load the DNN model
model = cv2.dnn.readNet(path_model + model_name)


if (model.empty()):
    print("Could not load the neural net! - Check path")


# Set backend and target to CUDA to use GPU
#model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

Q = cv_file.getNode('q').mat()
 
# Webcam
cap = cv2.VideoCapture(0)

# Read in the image
success, img = cap.read()


cv2.imshow('image', img)
cv2.waitKey(0)

img = downsample_image(img, 3)

cv2.imshow('image', img)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imgHeight, imgWidth, channels = img.shape


# Create Blob from Input Image
# MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
blob = cv2.dnn.blobFromImage(img, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

# MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
#blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

# Set input to the model
model.setInput(blob)

# Make forward pass in model
output = model.forward()

output = output[0,:,:]
output = cv2.resize(output, (imgWidth, imgHeight))

# Normalize the output
output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


plt.imshow(output,'gray')
plt.show()

# -------------------------------------------------------------------------------------

#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(output, Q, handleMissingValues=False)


#Get rid of points with value 0 (i.e no depth)
mask_map = output > output.min()

#Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = img[mask_map]



#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')
 


output_file = 'reconstructedMono.ply'

#Generate point cloud 
create_output(output_points, output_colors, output_file)



cap.release()
cv2.destroyAllWindows()
