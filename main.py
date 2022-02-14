import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import pixellib
from pixellib.tune_bg import alter_bg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import skimage.exposure

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


saved = load_model("save_ckp_frozen.h5")


class fashion_tools(object):
    def __init__(self,imageid,model,version=1.1):
        self.imageid = imageid
        self.model   = model
        self.version = version
        
    def get_dress(self,stack=False):
        """limited to top wear and full body dresses (wild and studio working)"""
        """takes input rgb----> return PNG"""
        name =  self.imageid
        file = cv2.imread(name)
        file = tf.image.resize_with_pad(file,target_height=512,target_width=512)
        rgb  = file.numpy()
        file = np.expand_dims(file,axis=0)/ 255.
        seq = self.model.predict(file)
        seq = seq[3][0,:,:,0]
        seq = np.expand_dims(seq,axis=-1)
        c1x = rgb*seq
        return c1x
        
        
    def get_patch(self):
        return None





#getting cloth from image
api  = fashion_tools('static/plzwork.jpg',saved)
img = api.get_dress(stack=True)
cv2.imwrite("static/cloth-old.jpg", img)

change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
output = change_bg.color_bg("static/cloth-old.jpg", colors = (255, 255, 255))

cv2.imwrite("static/cloth.jpg", output)


# load image and get dimensions 
img = cv2.imread("static/cloth.jpg")

# convert to hsv
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
L = lab[:,:,0]
A = lab[:,:,1]
B = lab[:,:,2]

# negate A
A = (255 - A)

# multiply negated A by B
nAB = 255 * (A/255) * (B/255)
nAB = np.clip((nAB), 0, 255)
nAB = np.uint8(nAB)


# threshold using inRange
range1 = 100
range2 = 160
mask = cv2.inRange(nAB,range1,range2)
mask = 255 - mask

# apply morphology opening to mask
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# antialias mask
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(0,255))

# put white where ever the mask is zero
result = img.copy()
result[mask==0] = (255,255,255)

# write result to disk
cv2.imwrite("static/bettercloth.jpg", result)




#print(img)
# plt.imshow(img)
# plt.show()

# -----------------------------------------------
#replacing clothing in person.jpg with cloth.jpg
# from Model import Model
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import time
# import cv2
# from matplotlib.image import imread

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model = Model("checkpoints/jpp.pb",
#               "checkpoints/gmm.pth",
#               "checkpoints/tom.pth",
#               use_cuda=False)

# change_bg = alter_bg()
# change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
# output = change_bg.color_bg("static/person-old.jpg", colors = (255, 255, 255))
# cv2.imwrite("static/person.jpg", output)

# img = Image.open("static/person.jpg")
# w, h = img.size
# img = img.resize((192, 256))
# img = np.array(img)
# plt.imshow(img)
# plt.show()

# c_img = Image.open("static/cloth.jpg")
# c_img = c_img.resize((192, 256))
# #c_img = c_img.convert('RGB')
# c_img = np.array(c_img)
# plt.imshow(c_img)
# plt.show()

# start = time.time()
# result,trusts = model.predict(img, c_img, need_pre=False,check_dirty=True)
# if result is not None:
#     end = time.time()
#     print("time:"+str(end-start))
#     print("Confidence"+str(trusts))
#     #result = cv2.resize(result, (w, h))
#     plt.imshow(result)
#     cv2.imwrite("static/final.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
#     plt.show()
