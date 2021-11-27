# %%
import PIL
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img_fullpath = '/home/brian/Dropbox/SideProjects/20211120_PetFinder_Kaggle/kaggle_petfinder/data/train/0a3ee4eb8d591d0cd5f3206ac0c5acd0.jpg'
# %%
img = PIL.Image.open(img_fullpath) # Reads in PIL object
img

# %%
img = cv2.imread(img_fullpath) # Reads in BGR, np.ndarray, dtype=uint8
img

# %%
img = mpimg.imread(img_fullpath) # Reads in RGB, np.ndarray, dtype=uint8
img

# %%
img = plt.imread(img_fullpath) # Reads in RGB, np.ndarray, dtype=uint8
img
