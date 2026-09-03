
#pragma once
#include "ofMesh.h"
#include "ofRectangle.h"
#include "ofxRtmHand.h"
#include "ofxRtmSkeletonSpec.h"

class ofxRtmPose {
public:
	
	enum PoseType:int {
		WHOLE_BODY_133=0,
		BODY_26, // HALPE
		BODY_17, // COCO
		HAND_2D_21
	};
	
	static std::string sGetPoseTypeName( const PoseType& aPoseType );
	
	struct Box {
		ofRectangle srcRect, outRect;
		float score = 0;
		int labelIndex = 0;
	};
	
	void drawWithLinkColors(float athresh);
	void draw(float athresh);
	
	void updateFromPose( const ofxRtmPose& apose );
	void updateFromPoseWithSmoothing( const ofxRtmPose& apose, float aSmoothing );
	
	PoseType getPoseType() { return mPoseType;}
	
	std::vector<glm::vec2> keypoints;
	std::vector<float> scores;
	float score = 0.0f;// total keypoint score
//	ofRectangle scaledRect;
	Box box;
	
	float getScore() const;
	ofRectangle getBoundingBox() const;
	
	int getKeypointIndex( const std::string& aname );
	glm::vec2 getKeypointPos( const std::string& aname );
	glm::vec2 getKeypointPos( const int& aindex );
	
	float getKeypointScore( const std::string& aname );
	float getKeypointScore( const int& aindex );
	
	bool isKeypointValid( const std::string& aname, float athreash );
	bool isKeypointValid( int aindex, float athresh );
	bool isKeypointValid( const glm::vec2& akeypoint );
	static bool sIsKeypointValid( const glm::vec2& akeypoint );
	
	// now lets try to get the hands out of here //
	ofxRtmHand& getLeftHand() {mLeftHand.handed = ofxRtmHand::LEFT;  return mLeftHand; }
	ofxRtmHand& getRightHand() {mRightHand.handed = ofxRtmHand::RIGHT; return mRightHand; }
	
	std::vector<int> getLeftHandIndices();
	std::vector<int> getRightHandIndices();
	bool hasValidHand( bool abLeft, float athresh );
	
	std::vector<glm::vec2> getHandKeypoints( bool abLeft );
	std::vector<float> getHandScores( bool abLeft );
	
	bool supportsHands() { return (mPoseType == WHOLE_BODY_133 || mPoseType == HAND_2D_21);}
	
protected:
	void configSpec();
	void _updateHands();
	
	template<typename T>
	std::vector<T> sliceVectorInclusive( const std::vector<T>& avalues, std::vector<int> aindices ) {
		if( aindices.size() < 1 ) {
			std::vector<T> kps;
			ofLogWarning("ofxRtmPose") << "sliceVectorInclusive " << sGetPoseTypeName(mPoseType) << " does not have aindices.";
			return kps;
		}
		
		// clamp indices into valid range
		int startIndex = std::clamp(aindices[0], 0, (int)avalues.size() - 1);
		int endIndex   = std::clamp(aindices.back(),   0, (int)avalues.size() - 1);
		// end iterator is exclusive, so +1 to include endIndex
		return std::vector<T>(avalues.begin() + startIndex,
							avalues.begin() + endIndex + 1 );
	}
	
	friend class ofxRtmPoseDetector;
//	friend class ofxRtmoPoseDetector;
	
//	struct LinkGroup {
//		std::vector<std::pair<int,int>> links;
//		ofFloatColor color;
//	};
	
//	struct SkeletonSpec {
//		std::string name;                                   // e.g. "wholebody133", "hand21"
//		std::vector<std::string> names;                     // size = #keypoints
//		std::vector<std::pair<int,int>> links;              // all linkes
//		// Optional named edge-groups so you can toggle sets at runtime (e.g. "body","face","left_hand")
//		std::unordered_map<std::string,LinkGroup> linkGroups;
//		std::unordered_map<std::string,int> name_to_idx;    // string -> index
//	};
	
	ofxRtmSkeletonSpec* mSkelSpecPtr = nullptr;
	PoseType mPoseType = WHOLE_BODY_133;
	// called from another class to guess the type of skeleton spec from number of keypoints.
	
	
//	static std::unordered_map<std::string, SkeletonSpec> sPoseRegistry;
//	
//	static  void sRegisterSkeletonSpec(const SkeletonSpec& s);
//	static SkeletonSpec* sGetSkeletonSpec(const std::string& name);
	
//	SkeletonSpec getWholebody133Spec();
//	SkeletonSpec getHand21Spec(const std::string& aname);
//	SkeletonSpec getHalpe26Spec();
//	SkeletonSpec getCoco17();
	
	ofxRtmHand mLeftHand, mRightHand;
	
//	int kp_index(const SkeletonSpec& spec, const std::string& name);
};
