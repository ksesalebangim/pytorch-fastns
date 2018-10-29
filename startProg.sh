#!/bin/sh

# Turn the brightness to 0
SCREEN_NAME=HDMI-0


#xrandr --output $SCREEN_NAME --brightness 0.9
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.8
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.7
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.6
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.5
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.4
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.3
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.2
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0

# kill send q to old window (called frame) to close it
wmctrl -a frame && sleep .8 && xdotool key q

case "$1" in

1) echo "1"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/40_neon_lion_by_theferraci-d6y30q0.jpg.trained_1_1_15_size_512_dataset_coco.model &
;;
2) echo "2"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/72_lion_head_in_geometric_design_vector_6825642.jpg.trained_1_1_30_size_512_dataset_coco.model &
;;
3) echo "3"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/483_drawings-of-house-plans-architecture-architectural-building-2d-autocad-1024x819.jpg.trained_1_1_31_size_512_dataset_coco.model &
;;
4) echo "4"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/79_poly_dog.jpg.trained_5_1_30_size_512_dataset_coco.model &
;;
5) echo "5"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
;;
a) echo "a"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/427_adriftII-painting48x36-webres.jpg.trained.jpg.trained_1_1_31_size_512_dataset_coco.model &
;;
b) echo "b"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/45_4e5348785137b686683ff309991d6fd6.jpg.trained_1_1_15_size_512_dataset_coco.model &
;;
c) echo "c"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/40_neon_lion_by_theferraci-d6y30q0.jpg.trained_1_1_15_size_512_dataset_coco.model &
;;
d) echo "d"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/456_Andy-Kehoe-Levine-AM-28.jpg.trained.jpg.trained_1_1_31_size_512_dataset_coco.model &
;;
e) echo "e"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/408_tipperFB.jpg.trained.jpg.trained_1_1_31_size_512_dataset_coco.model &
;;
f) echo "f"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/439_ovm-fungi_detail.jpg.trained.jpg.trained_1_1_31_size_512_dataset_coco.model &
;;
g) echo "g"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/53_trippy-eyes-alex-grey.jpg.trained_1_1_15_size_512_dataset_coco.model &
;;
h) echo "h"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/54_organic_by_suicidebysafetypin-d4w93sw.jpg.trained_1_1_15_size_512_dataset_coco.model &
;;
i) echo "i"
cd /media/midburn/benson2/pytorch/pytorch-fastns/
python3 neural_style/webcam.py --cuda 1 --model saved-models/24_lc-2-1.jpg.trained_1_1_18_size_512_dataset_coco.model &
;;
esac 

# Giving the app 7 seconds to load
#sleep 6

# Turn the brightness to 1
SCREEN_NAME=HDMI-0

#xrandr --output $SCREEN_NAME --brightness 0.1
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.2
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.3
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.4
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.5
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.6
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.7
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.8
#sleep .1
#xrandr --output $SCREEN_NAME --brightness 0.9
#sleep .1
xrandr --output $SCREEN_NAME --brightness 1



