# Installation

First clone with submodules installed:

 `git clone --recurse-submodules https://github.com/simonalford42/abstraction`

If you've already cloned, try

 `git submodule update --init --recursive`

to get the submodules added.

Then install the requirements:

 `pip install -r requirements.txt`

and
 
 `pip install -r yolov7/requirements.txt`

Lastly, download the yolov7 pose estimation model: 

 `curl -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt -o yolov7-w6-pose.pt`

# Usage
This is a tool to estimate shooting form consistency. The steps are
1. Pass in a video
2. Indicate the first frame of the first shot, and optionally the number of shots taken.
3. Receive an estimate of shooting consistency.

The consistency score is given out of 100. Perfect consistency is 100, and perfect inconsistency is 0. Perfect inconsistency is calculated by taking random poses, so in theory worse than zero is also possible. 

Some notes:
- It takes time and memory to run yolov7 on the images of the video. If needed, the batch size can be decreased to reduce memory burden.
- When prompted, input the first frame where the first shot begins. Looking at the created annotated frames folder is usedful for this.
- It is optional to provide the number of shots in the video. If no number is provided, the detecter will attempt to estimate the number of shots based on the cyclical change in pose.
