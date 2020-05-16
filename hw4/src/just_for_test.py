import utils
import parser
import models
import os

full_video_path = '../hw4_data/FullLengthVideos/videos/valid/'
category_list = sorted(os.listdir(os.path.join(full_video_path)))
print(category_list)
