#!/bin/sh
# Turn the brightness to 0
SCREEN_NAME=HDMI-0

xrandr --output $SCREEN_NAME --brightness 0.9
sleep .1
xrandr --output $SCREEN_NAME --brightness 0.8
sleep .1
xrandr --output $SCREEN_NAME --brightness 0.7
sleep .1
xrandr --output $SCREEN_NAME --brightness 0.6
sleep .1
xrandr --output $SCREEN_NAME --brightness 0.5
sleep .1
xrandr --output $SCREEN_NAME --brightness 0.4
sleep .1
xrandr --output $SCREEN_NAME --brightness 0.3
sleep .1
xrandr --output $SCREEN_NAME --brightness 0.2
sleep .1
xrandr --output $SCREEN_NAME --brightness 0.01

