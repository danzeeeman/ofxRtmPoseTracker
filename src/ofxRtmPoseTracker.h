
#pragma once
#include "ofxRtmPoseDetector.h"
#include "ofParameter.h"
#include "ofThread.h"


class ofxRtmTrackedHand : public ofxRtmHand {
public:
	int ID = 0;
	float age = 0.f;
	int poseID = -1;
	
	float timeNotFound = 0.f;
	bool bRemove = false;
	bool bFoundThisFrame = false;
};

class ofxRtmTrackedPose : public ofxRtmPose {
public:
	float age = 0.0f;
	// we create a new tracked pose when a pose is found
	bool bFoundThisFrame = true;
	float timeNotFound = 0.f;
	bool bRemove = false;
	uint64_t ID = 0;
	std::vector<glm::vec2> prevKeypoints;
	
	std::vector< std::shared_ptr<ofxRtmTrackedHand> > hands;
};

class ofxRtmPoseTracker : public ofThread {
public:
	
	class Settings {
	public:
		of::filesystem::path detectorModelPath;
		// set only poseModelPath for loading RTMO
		of::filesystem::path poseModelPath;
		ofLogLevel logLevel = OF_LOG_VERBOSE;
	};
	
	ofxRtmPoseTracker();
	~ofxRtmPoseTracker();
	
	ofParameterGroup& getParams( std::string aname );
	
	bool setup( ofxRtmPoseTracker::Settings asettings );
	void clear();
	
	void start();
	void stop();
	
	
	void update( ofEventArgs& args );
	
	bool hasNewData();
	bool run(ofPixels& apixels);
	
	void draw();
	void draw( const ofRectangle& arect );
	void draw(float ax, float ay, float awidth, float aheight);
	
	void drawHand( std::shared_ptr<ofxRtmTrackedHand> ahand, const ofRectangle& arect );
	
	float getMinKpScoreToTrack() { return mMinKeypointScoreToTrack; }
	
	std::vector<std::shared_ptr<ofxRtmTrackedPose>>& getPoses() { return mPoses;}
	std::vector<std::shared_ptr<ofxRtmTrackedHand>>& getHands() { return mHands;}
	std::vector<ofxRtmPose> getInPoses() { return mInPoses;}

	std::shared_ptr<ofxRtmPoseDetector> getRtmPoseDetector() {return mRtmPoseDetector;}
	
	ofRectangle getOutRect() {
		if( mRtmPoseDetector ) {
			return mRtmPoseDetector->getOutRect();
		}
		return ofRectangle();
 	}
	
	ofRectangle getSrcRect() {
		if( mRtmPoseDetector ) {
			return mRtmPoseDetector->getSrcRect();
		}
		return ofRectangle();
	}
	
	void setOutRect( const ofRectangle& arect ) {
		if( mRtmPoseDetector ) {
			mRtmPoseDetector->setOutRect(arect);
		}
	}
	
	ofFpsCounter& getFpsCounter() { return mFpsCounter; }
	double getFps() { return mFpsCounter.getFps(); }
	
	void setEnableUpdateListeners( bool ab ) {mBEnableUpdateListeners=ab;}
	
protected:
	float _poseMatchCost(ofxRtmPose& det, shared_ptr<ofxRtmTrackedPose>& tracked, const ofRectangle& aFrameRect);
	float _handMatchCost(const ofxRtmHand& det, const shared_ptr<ofxRtmTrackedHand>& tracked, const ofRectangle& aFrameRect);
	float _handPoseMatchCost(const shared_ptr<ofxRtmTrackedPose> apose,
							 const shared_ptr<ofxRtmTrackedHand>& ahand, const ofRectangle& aFrameRect);
	std::vector<ofxRtmHand> _suppressOverlappingHands(const std::vector<ofxRtmHand>& incoming, float iouThreshold = 0.4f);
	std::vector<int> _hungarianMatch( const std::vector<std::vector<float>>& costMatrix, float maxCostThreshold);
	
	bool hasPendingPixels() {
		if(tryLock()) {
			bool v = mBHasPendingPixels;
			unlock();
			return v;
		}
		return false;
	}
	
	void threadedFunction() override;
	
	void _matchPoses( std::vector<ofxRtmPose>& aIncomingPoses, std::vector<std::shared_ptr<ofxRtmTrackedPose>>& aTrackedPoses );
	void _matchHands( std::vector<ofxRtmPose>& aIncomingPoses, std::vector<std::shared_ptr<ofxRtmTrackedHand>>& aTrackedHands );
	void _assocTrackedHandsWithTrackedPoses(std::vector<std::shared_ptr<ofxRtmTrackedPose>> aPoses,
											std::vector<std::shared_ptr<ofxRtmTrackedHand>> aHands);
	
	void _addAppListeners();
	void _removeAppListeners();
	
	ofParameterGroup mParams;
	
	ofxRtmPoseTracker::Settings mSettings;
	std::shared_ptr<ofxRtmPoseDetector> mRtmPoseDetector;
	
	bool mBLoaded = false;
	
	ofFpsCounter mFpsCounter;
	uint64_t mIdCounter = 0;
	uint64_t mIdCounterHand = 0;
	
	ofParameter<float> mPosSmoothing, mOverlapThreshN;
	ofParameter<float> mBoxThresh, mDrawThresh, mDurNotFoundToKill, mMinKeypointScoreToTrack;//, mMaxKeypointDiffToMatch;
	ofParameter<float> mHandMatchCostThresh, mPoseMatchCostThresh, mHandOverlapThresh;
	
	ofParameter<bool> mBRemoveOverlappingPoses, mBRemoveOverlappingHands;
	ofParameter<bool> mBDrawBoxes, mBDrawJoints, mBColorById, mBDrawSrcPoses;
	ofParameter<bool> mBDrawPoses, mBDrawHands;
	
	ofParameter<float> mMinPoseSize, mMinHandSize;
	
	std::vector<ofxRtmPose> mThreadedPoses;
	std::vector<ofxRtmPose> mInPoses;
	
	std::vector<std::shared_ptr<ofxRtmTrackedPose>> mPoses;
	std::vector<std::shared_ptr<ofxRtmTrackedHand>> mHands;
	
	bool mBHasAppListeners=false;
	bool mBEnableUpdateListeners = true;
	
	
	ofPixels mPixelsPending;
	std::atomic<bool> mBHasNewThreadedPoses = false;
	std::mutex mCvMutex, mCopyMutex;
	std::condition_variable mCondition;
	bool mBHasPendingPixels = false;
	bool mBWakeThread = false;
	
	bool mBHasNewData = false;
	
};

