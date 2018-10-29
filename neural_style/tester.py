import operator
from PIL import Image
from PIL import ImageDraw
import cv2
cam = cv2.VideoCapture(0)
ret_val, original1 = cam.read()
img2=cv2.flip(original1,1)
ret_val, original = cam.read()
img1=cv2.flip(original,1)




img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img1 = Image.fromarray(img1)
# suppose img2 is to be shifted by `shift` amount 
shift = (50, 60)
print(img2.size)
# compute the size of the panorama
nw, nh = map(max, map(operator.add, img1.size, shift), img1.size)

# paste img1 on top of img2
newimg1 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
newimg1.paste(img1, shift)
newimg1.paste(img1, (0, 0))

# paste img2 on top of img1
newimg2 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
newimg2.paste(img1, (0, 0))
newimg2.paste(img1, shift)

# blend with alpha=0.5
result = Image.blend(newimg1, newimg2, alpha=0.5)
