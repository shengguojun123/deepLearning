import cnn_utils
import matplotlib.pyplot as plt
from PIL import Image


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()

my_image1 = X_train_orig[2]
img = Image.fromarray(my_image1,'RGB')
img.save('./my.png')
plt.imshow(img)
plt.show()
