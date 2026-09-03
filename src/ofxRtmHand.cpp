
#include "ofxRtmHand.h"
#include "ofxRtmPose.h"

using namespace std;

//---------------------------------------
void ofxRtmHand::updateFromHand( const ofxRtmHand& ahand ) {
	keypoints = ahand.keypoints;
	scores = ahand.scores;
	_checkSkel();
}

//---------------------------------------
void ofxRtmHand::updateFromHandWithSmoothing( const ofxRtmHand& ahand, float aSmoothing ) {
	scores = ahand.scores;
	
	size_t numKps = ahand.keypoints.size();
	if( numKps != keypoints.size() ) {
		keypoints = ahand.keypoints;
		return;
	}
	
	for( std::size_t i = 0; i < numKps; i++ ){
		auto & kp = keypoints[i];
		auto & kpNew = ahand.keypoints[i];
		
		bool kpValid = ofxRtmPose::sIsKeypointValid(kp);
		bool kpNewValid = ofxRtmPose::sIsKeypointValid(kpNew);
		
		if( !kpValid && kpNewValid ) {
			kp = kpNew;
		}
		kp = glm::mix(kp, kpNew, 1.0f-aSmoothing );
	}
	
	_checkSkel();
}

//---------------------------------------
bool ofxRtmHand::isKeypointValid( int aindex, float athresh ) {
	if (aindex < 0 || aindex >= (int)keypoints.size()) return false;
	const auto& p = keypoints[aindex];
	if( !ofxRtmPose::sIsKeypointValid( p )) { return false; }
	if (aindex >= (int)scores.size()) return false;
	if (scores[aindex] < athresh) return false;
	return true;
}

//---------------------------------------
bool ofxRtmHand::isKeypointValid( int aindex, float athresh ) const {
	if (aindex < 0 || aindex >= (int)keypoints.size()) return false;
	const auto& p = keypoints[aindex];
	if( !ofxRtmPose::sIsKeypointValid( p )) { return false; }
	if (aindex >= (int)scores.size()) return false;
	if (scores[aindex] < athresh) return false;
	return true;
}

//---------------------------------------
glm::vec2 ofxRtmHand::getPalmPos() {
	auto palmPos = glm::vec2(0.f, 0.f);
	if( keypoints.size() > 17 ) {
		float numPs = 0.f;
		vector<int> pindices = {0, 5, 9, 13, 17 };
		for( int k = 0; k < pindices.size(); k++ ) {
			palmPos += keypoints[pindices[k]];
			numPs += 1.f;
		}
		if( numPs > 1.f ) {
			palmPos /= numPs;
		}
	}
	return palmPos;
}

//---------------------------------------
glm::vec2 ofxRtmHand::getPalmPos() const {
	auto palmPos = glm::vec2(0.f, 0.f);
	if( keypoints.size() > 17 ) {
		float numPs = 0.f;
		vector<int> pindices = {0, 5, 9, 13, 17 };
		for( int k = 0; k < pindices.size(); k++ ) {
			palmPos += keypoints[pindices[k]];
			numPs += 1.f;
		}
		if( numPs > 1.f ) {
			palmPos /= numPs;
		}
	}
	return palmPos;
}

//---------------------------------------
ofRectangle ofxRtmHand::getBoundingBox() const {
    if (keypoints.empty()) return {};

    float minX = keypoints[0].x, maxX = keypoints[0].x;
    float minY = keypoints[0].y, maxY = keypoints[0].y;

    for (size_t i = 1; i < keypoints.size(); i++) {
        if (keypoints[i].x < minX) minX = keypoints[i].x;
        if (keypoints[i].x > maxX) maxX = keypoints[i].x;
        if (keypoints[i].y < minY) minY = keypoints[i].y;
        if (keypoints[i].y > maxY) maxY = keypoints[i].y;
    }

    return ofRectangle(minX, minY, maxX - minX, maxY - minY);
}

//---------------------------------------
float ofxRtmHand::getScore() {
	float tscore = 0.f;
	for( auto& sc : scores ) {
		tscore += sc;
	}
	return tscore;
}

//---------------------------------------
float ofxRtmHand::getScore() const {
	float tscore = 0.f;
	for( auto& sc : scores ) {
		tscore += sc;
	}
	return tscore;
}

//---------------------------------------
void ofxRtmHand::draw(float athresh) {
	_checkSkel();
	
	if( mSkelSpecPtr ) {
		ofMesh linesMesh;
		linesMesh.setMode(OF_PRIMITIVE_LINES);
		for( auto& skelG : mSkelSpecPtr->linkGroups ) {
			auto& linkG = skelG.second;
			auto& links = linkG.links;
			for( auto& link : links ) {
				int a=link.first, b=link.second;
				if( isKeypointValid(a,athresh) && isKeypointValid(b,athresh) ) {
					linesMesh.addVertex(glm::vec3(keypoints[a], 0.0f));
					linesMesh.addVertex(glm::vec3(keypoints[b], 0.0f));
				}
			}
		}
		linesMesh.draw();
	} else {
		ofLogWarning("ofxRtmHand::draw") << "no skel spec ptr!!! | " << ofGetFrameNum();
	}
}

//---------------------------------------
float ofxRtmHand::fingerLength(const std::vector<glm::vec2>& kps, int start, int end) {
	float len = 0;
	for (int i = start; i < end; i++)
		len += glm::distance(kps[i], kps[i+1]);
	return len;
}

//---------------------------------------
bool ofxRtmHand::hasPlausibleFingerLengths(float aMinHandSize) {
	const auto& kps = keypoints;
	if (kps.size() < 21) return false;
	
	float handSize = glm::distance(kps[0], kps[9]); // wrist to middle MCP
	if(handSize < aMinHandSize) {return false;} // degenerate, less than a pixel
	
	float thumbLen  = fingerLength(kps, 1,  4);
	float indexLen  = fingerLength(kps, 5,  8);
	float middleLen = fingerLength(kps, 9,  12);
	float ringLen   = fingerLength(kps, 13, 16);
	float pinkyLen  = fingerLength(kps, 17, 20);
	
	// Each finger should be between 0.3x and 2.5x the wrist-to-MCP distance
	auto inRange = [&](float len) {
		return len > handSize * 0.3f && len < handSize * 2.5f;
	};
	
	return inRange(thumbLen) && inRange(indexLen) && inRange(middleLen) && inRange(ringLen)  && inRange(pinkyLen);
}

//---------------------------------------
void ofxRtmHand::_checkSkel() {
	if( !mSkelSpecPtr ) {
		auto pname = ofxRtmPose::sGetPoseTypeName( ofxRtmPose::HAND_2D_21 );
		if( !ofxRtmSkeletonSpecRegistry::sHasSkeletonSpec(pname)) {
			ofxRtmSkeletonSpec tspec;
			ofxRtmSkeletonSpecRegistry::sRegisterSkeletonSpec(pname, tspec.getHand21Spec() );
		}
		mSkelSpecPtr = ofxRtmSkeletonSpecRegistry::sGetSkeletonSpec(pname);
		
	}
}
