
#pragma once
#include "ofMesh.h"

class ofxRtmSkeletonSpec {
public:
	
	struct LinkGroup {
		std::vector<std::pair<int,int>> links;
		ofFloatColor color;
	};
	
	std::string name;                                   // e.g. "wholebody133", "hand21"
	std::vector<std::string> names;                     // size = #keypoints
	std::vector<std::pair<int,int>> links;              // all linkes
														// Optional named edge-groups so you can toggle sets at runtime (e.g. "body","face","left_hand")
	std::unordered_map<std::string,LinkGroup> linkGroups;
	std::unordered_map<std::string,int> name_to_idx;    // string -> index
	
	
	ofxRtmSkeletonSpec getWholebody133Spec();
	ofxRtmSkeletonSpec getHand21Spec();
	ofxRtmSkeletonSpec getHalpe26Spec();
	ofxRtmSkeletonSpec getCoco17();
};

class ofxRtmSkeletonSpecRegistry {
public:
	
	static void sRegisterSkeletonSpec(const std::string& aname, const ofxRtmSkeletonSpec& s);
	static bool sHasSkeletonSpec(const std::string& aname);
	static ofxRtmSkeletonSpec* sGetSkeletonSpec(const std::string& aname);
	
protected:
	static std::unordered_map<std::string, ofxRtmSkeletonSpec> sSkelRegistry;
};
