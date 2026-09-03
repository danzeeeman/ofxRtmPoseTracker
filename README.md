# ofxRtmPoseTracker
Wrapper for running inference on the RTM Pose onnx models. Requires ofxOnnx and ofxOpenCv.
ofxOnnx can be found [here](https://github.com/design-io/ofxOnnx)

### MMPose
The addon leverages the [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) and [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) onnx models from MMPose.

### Tracking
The addon tracks poses between frames with parameters to tune the confidence and matching thresholds. Run the inference in a separate thread calling `start();`

See the included example for more implementation details. 

<img width="1101" height="656" alt="Screenshot 2026-09-03 at 5 30 55 PM" src="https://github.com/user-attachments/assets/ad19bdf0-01ba-4be8-a264-58d8c7c69ff2" />


### References:
 - https://github.com/open-mmlab/mmpose
 - https://github.com/Tau-J/rtmlib/tree/main
 - https://github.com/HW140701/RTMPose-Deploy/tree/main/Windows/OnnxRumtime-CPU/src/RTMPoseOnnxRuntime
