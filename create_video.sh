#!/bin/bash  
  
# Change directory to the frames directory  
cd frames  
  
# Use FFmpeg to combine the images into a video  
ffmpeg -framerate 60 -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow ../trajectories.mp4  
  
# Return to the original directory  
cd ..  
  
echo "Video created."  
