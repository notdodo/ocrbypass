#!/usr/bin/env zsh
#

rm -r ./trimmed/
rm -r ./improved/
rm -r ./words/
mkdir -p ./trimmed/
mkdir -p ./improved/
mkdir -p ./words/

# INPUT: captchas folder
for i in ./*; do
  IMG_NAME="$(basename ${i})"
  # typeset -i WIDTH
  # WIDTH=$(magick identify -format "%[fx:w]" 33pf7)

  # Trim the image
  convert "${i}" -fuzz 6% -trim +repage "./trimmed/${IMG_NAME}.png"
  # Increase contrast: binarize
  convert -threshold 70% "./trimmed/${IMG_NAME}.png" "./improved/${IMG_NAME}.png"
  # Split the image into the 5 five chars of the captcha
  convert "./improved/${IMG_NAME}.png" -crop 5x1@ +repage +adjoin ./words/${IMG_NAME}-%d.png
done

# Fix all images to the same size 30x40
for i in ./words/*; do
  IMG_NAME="$(basename ${i})"
  convert -size 30x40 xc:white "${i}" -gravity center -composite "${i}"
done
