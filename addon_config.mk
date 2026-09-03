meta:
    ADDON_NAME = ofxRtmPoseTracker
    ADDON_DESCRIPTION =
    ADDON_AUTHOR = Nick Hardeman
    ADDON_TAGS = "rtmpose" "rtmo" "machine learning" "inference" "pose tracker"
    ADDON_URL =

common:
    # defines that will be passed to the compiler when including this addon
	ADDON_DEFINES = OF_ADDON_HAS_OFX_RTM_POSE_TRACKER
	
	ADDON_DEPENDENCIES = ofxOnnx
	ADDON_DEPENDENCIES += ofxOpenCv
	
osx:
    ADDON_LDFLAGS =
    ADDON_CFLAGS =

linux64:
    ADDON_LDFLAGS =
    ADDON_CFLAGS =
