

#pragma once
#include "ofxOnnx.h"
#include "ofxOpenCv.h"
#include "ofxRtmPose.h"

// References:
// RTM Pose https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
// https://github.com/Dominic23331/rtmpose_tensorrt/tree/master
// https://github.com/Tau-J/rtmlib/tree/main
// https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/examples/RTMPose-Deploy/Windows/OnnxRumtime-CPU/src/RTMPoseOnnxRuntime

class ofxRtmPoseDetector {
public:
	// overlapping computations
	static float sComputeIoU(const ofRectangle& a, const ofRectangle& b);
	
	ofxRtmPoseDetector() = default;
	~ofxRtmPoseDetector() {_removeAppListeners();}
	
	// LOADS RTMO MODEL
	bool load(std::filesystem::path aOnnxModelPath);
	bool load(std::filesystem::path aOnnxModelPath, ofLogLevel aLogLevel );
	bool loadRtmo( ofxOnnx::Settings aOnnxSettings );
	// LOAD TWO PASS MODELS; detector and model
	bool load(std::filesystem::path aDetModelPath, std::filesystem::path aPoseModelPath );
	bool load(std::filesystem::path aDetModelPath, std::filesystem::path aPoseModelPath, ofLogLevel aLogLevel );
	bool loadDetector(ofxOnnx::Settings aOnnxSettings );
	bool loadPose(ofxOnnx::Settings aOnnxSettings );
	
	void update( ofEventArgs& args );
	bool run(ofPixels& apixels);
	void draw();
	
	bool hasNewData() { return mBHasNewData;}
	
	void setDrawThresh( float athresh ) { mDrawThresh = athresh;}
	float getDrawThresh() { return mDrawThresh;}
	
	void setBoxThresh( float athresh ) { mBoxThresh = athresh; }
	float getBoxThresh() { return mBoxThresh; }
	
	void setAddDetectBoxIfNonDetected( bool ab ) { mBAddBoxIfNoneDetected = ab; }
	
	void setRemoveOverlappingPoses( bool ab ) { mBRemoveOverlappingPoses = ab; }
	bool getRemoveOverlappingPoses() { return mBRemoveOverlappingPoses; }
	void setOverlapToRemoveThresh( float aThreshNormalized ) { mPoseOverlapThreshToRemove = aThreshNormalized; }
	float getOverlapToRemoveThresh() { return mPoseOverlapThreshToRemove; }
	
	std::vector<ofxRtmPose::Box>& getBoxes() { return  mBoxes; }
	std::vector<ofxRtmPose>& getPoses() {return  mPoses;}
	
	void setOutRect(const ofRectangle& arect) { mOutRect = arect; }
	
	const ofRectangle& getSrcRect() { return mSrcRect; }
	const ofRectangle& getOutRect() { return mOutRect; }
	const ofRectangle & getOutRect() const { return mOutRect; }
		
protected:
	struct TBox {
		float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
		float score = 0;
	};
	
	// returns scale and offset
	std::tuple<float, glm::vec2> preprocess(const cv::Mat & frame_rgb, int aw, int ah );
	void hwc_to_nchw_rgb_u8_to_f32(const cv::Mat & img, float * outNCHW);
	
	float iou(const TBox& a, const TBox& b);
	std::vector<int> nms(const std::vector<TBox> & boxes, float nms_thr, float score_thr);
	
	void postprocess_rtmo(const Ort::Value & dets_val,
					 const Ort::Value & kps_val,
					 float ratio,
					 std::optional<float> nms_thr,
						  std::optional<float> score_thr);
	void postProcess(const cv::Mat & frame_rgb, std::vector<ofxRtmPose::Box>& aboxes, std::vector<ofxRtmPose>& aresults );
	
	std::vector<ofxRtmPose> _removeOverlappingPoses( const std::vector<ofxRtmPose>& aposes, float aiouThreshold );
	
	void mapPoseToOutRect( ofxRtmPose& apose );
	
	void _onExit( ofEventArgs& args );
	void _addAppListeners();
	void _removeAppListeners();
	
	
	
	
	std::array<int64_t, 4> mDetInputShape;
	std::vector<float> mDetInputBuffer; // reused every frame
	
	ofxOnnx mOnnxDet, mOnnxPose;
	cv::Mat mInCvMat;
	cv::Mat mCvPadded, mCvResized;
	
	ofRectangle mSrcRect, mOutRect;
	
	ofPixels mDetPixels;
	ofTexture mDetTex;
	bool mBUseInputTexture = false;
	bool mBDetTexDirty = false;
	
	std::vector<ofxRtmPose::Box> mBoxes;
	std::vector<ofxRtmPose> mPoses;
	
	bool mBRtmo = false;
	
	bool mBAddBoxIfNoneDetected = false;
	
	float mDrawThresh = 0.5f;
	float mBoxThresh = 0.5f;
	
	bool mBHasNewData = false;
	bool mBHasAppListeners=false;
	bool mBRemoveOverlappingPoses = true;
	
	float mPoseOverlapThreshToRemove = 0.5f;
};
