import sys

from ssd import SSD300 as SSD

from videotest import VideoTest

sys.path.append("..")

input_shape = (300, 300, 3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('../weights_SSD300.hdf5')

vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
video_path = 0
if len(sys.argv) > 1:
    video_path = sys.argv[1]
vid_test.run(video_path)
