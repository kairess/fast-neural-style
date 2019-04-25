import cv2
import numpy as np
import glob, sys, os

MEAN_VALUE = [103.939, 116.779, 123.680]

models = sorted(glob.glob('models/*/*.t7'))
model_idx = 0
net = cv2.dnn.readNetFromTorch(models[model_idx])

video_path = sys.argv[1]
filename = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
  ret, img_ori = cap.read()
  if not ret:
    break

  img = cv2.resize(img_ori, (640, int(img_ori.shape[0] / img_ori.shape[1] * 640)))
  h, w, _ = img.shape

  blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(w, h), mean=MEAN_VALUE, swapRB=False, crop=False)
  net.setInput(blob)
  output = net.forward()
  
  output = output.squeeze().transpose((1, 2, 0))
  
  output += MEAN_VALUE
  output = np.clip(output, 0, 255)
  output = output.astype(np.uint8)

  output = cv2.resize(output, (img_ori.shape[1], img_ori.shape[0]))

  cv2.imshow('output', output)

  key = cv2.waitKey(1)

  if key == 32: # load next model
    model_idx = (model_idx + 1) % len(models)

    net = cv2.dnn.readNetFromTorch(models[model_idx])
  
  if key == ord('q'):
    break
