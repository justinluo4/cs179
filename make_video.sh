#!/bin/bash

# Script to create a video from BMP frames using ffmpeg

# Frame rate for the output video
FRAMERATE=30

# Subdirectory for frames
FRAME_SUBDIR="build/frames_temp"

# Input pattern for BMP frames (e.g., frames_temp/frame_0000.bmp, ...)
# Ensure this matches the output of your C++ program.
INPUT_PATTERN="${FRAME_SUBDIR}/frame_%04d.bmp"

# Output video filename
OUTPUT_VIDEO="blackhole_lensing_animation.mp4"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found. Please install ffmpeg."
    echo "On Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg"
    echo "On macOS (using Homebrew): brew install ffmpeg"
    exit 1
fi

# Create the frame subdirectory if it doesn't exist
if [ ! -d "${FRAME_SUBDIR}" ]; then
    echo "Creating frame subdirectory: ${FRAME_SUBDIR}"
    mkdir -p "${FRAME_SUBDIR}"
    if [ $? -ne 0 ]; then
        echo "Failed to create directory ${FRAME_SUBDIR}. Exiting."
        exit 1
    fi
fi

# Check if there are any BMP frames to process in the subdirectory
# Note: This check needs to happen *after* the C++ program has run.
# The script will typically be run after frame generation.
if ! ls ${FRAME_SUBDIR}/frame_*.bmp 1> /dev/null 2>&1; then
    echo "No BMP frames found in '${FRAME_SUBDIR}/' matching the pattern 'frame_*.bmp'."
    echo "Please generate the frames first by running the C++ program."
    echo "(The C++ program should save frames into the '${FRAME_SUBDIR}' directory.)"
    exit 1
fi

echo "Found frames in '${FRAME_SUBDIR}/'. Creating video '${OUTPUT_VIDEO}' with framerate ${FRAMERATE}fps..."

# ffmpeg command:
# -framerate: Input framerate
# -i: Input file pattern (now includes subdirectory)
# -c:v libx264: Video codec (H.264, widely compatible)
# -pix_fmt yuv420p: Pixel format, good for compatibility with most players
# -y: Overwrite output file if it exists
ffmpeg -framerate ${FRAMERATE} -i "${INPUT_PATTERN}" -c:v libx264 -pix_fmt yuv420p -y "${OUTPUT_VIDEO}"

if [ $? -eq 0 ]; then
    echo "Video '${OUTPUT_VIDEO}' created successfully!"
else
    echo "ffmpeg command failed. Please check for errors."
fi

# Optional: Clean up the BMP frames and the subdirectory after creating the video
read -p "Do you want to delete the BMP frames and the '${FRAME_SUBDIR}' directory? (y/N): " choice
case "$choice" in 
  y|Y ) 
    echo "Deleting BMP frames and directory '${FRAME_SUBDIR}'..."
    rm -rf "${FRAME_SUBDIR}" # Use rm -rf to remove directory and its contents
    echo "Frames and directory deleted."
    ;;
  * ) 
    echo "Frames and directory '${FRAME_SUBDIR}' were not deleted."
    ;;
esac

echo "Done." 