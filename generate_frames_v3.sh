#!/bin/bash

# markers
red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
reset=`tput sgr0`

# functions
function join_by 
{ 
    local d=${1-} f=${2-}; 
    if shift 2; then 
        arr=("$@"); 
        printf %s "\.$f" "${arr[@]/#/$d}$"; 
    fi; 
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASETPATH="${SCRIPT_DIR}/dataset/$1"
if [ -z "$1" ]; then
    echo "dataset name is not provided"
    exit 1 
fi
# echo "$1"

if [ ! -d "$DATASETPATH" ]; then
    echo "folder $DATASETPATH does not exist"
    exit 1
fi

# echo "$DATASETPATH"



FORMATS=( "avi" "mov" "wmv" "mp4" )

# join_by "$|\." "${FORMATS[@]}"
query=$(join_by "$|\." "${FORMATS[@]}")
# echo ${query}
# VIDS=()
# VIDS=$(find $DATASETPATH -maxdepth 1 -type f | grep -iE ${query})
VIDS=$(find $DATASETPATH -type f | grep -iE ${query})
# len=${#VIDS[#]}
# echo $len

# GENERATE FRAMES
for vid_path in ${VIDS[@]}; do
    # 
    video_folder=$(dirname $vid_path)
    echo $video_folder

    # 
    video_frames_folder="${video_folder}/frames"
    # echo $video_frames_folder

    # create frames folder
	rm -rf $video_frames_folder
	mkdir -p $video_frames_folder
    
	echo "${video_frames_folder} ${green}created${reset}"

    # generate frames
	echo "${yellow}generating frames for ${vid_path} ...${reset}"
	ffmpeg -r 1 -i $vid_path -r 1 ${video_frames_folder}/$frame%06d.png
	if [ $? == 0 ]; then
		echo "${green}[DONE]${reset}"
	fi

done

