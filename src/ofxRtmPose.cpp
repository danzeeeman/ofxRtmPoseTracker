
#include "ofxRtmPose.h"
#include "ofGraphics.h"

//--------------------------------------------------------------
std::string ofxRtmPose::sGetPoseTypeName( const PoseType& aPoseType ) {
	if( aPoseType == HAND_2D_21 ) {
		return "Hand2D_21";
	} else if(aPoseType == BODY_26 ) {
		return "Body_Halpe_26";
	} else if( aPoseType == BODY_17 ) {
		return "Body_COCO_17";
	}
	return "WholeBody_133";
}

//---------------------------------------
float ofxRtmPose::getScore() const {
	float tscore = 0.f;
	for( auto& sc : scores ) {
		tscore += sc;
	}
	return tscore;
}

//---------------------------------------
ofRectangle ofxRtmPose::getBoundingBox() const {
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

// -------------------------------
int ofxRtmPose::getKeypointIndex( const std::string& aname ) {
	if( mSkelSpecPtr == nullptr ) return -1;
	if( mSkelSpecPtr->name_to_idx.empty() ) {return -1;}
	auto it = mSkelSpecPtr->name_to_idx.find(aname);
	return (it==mSkelSpecPtr->name_to_idx.end()) ? -1 : it->second;
}

// -------------------------------
glm::vec2 ofxRtmPose::getKeypointPos( const std::string& aname ) {
	return getKeypointPos(getKeypointIndex(aname));
}

// -------------------------------
glm::vec2 ofxRtmPose::getKeypointPos( const int& aindex ) {
	if (aindex < 0 || aindex >= (int)keypoints.size()) {
		return glm::vec2(0.f, 0.f);
	}
	const auto& p = keypoints[aindex];
	if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
		return glm::vec2(0.f, 0.f);
	}
	return p;
}

// -------------------------------
float ofxRtmPose::getKeypointScore( const std::string& aname ) {
	return getKeypointScore(getKeypointIndex(aname));
}

// -------------------------------
float ofxRtmPose::getKeypointScore( const int& aindex ) {
	if (aindex < 0 || aindex >= (int)scores.size()) return 0.0f;
	return scores[aindex];
}

//---------------------------------------
bool ofxRtmPose::isKeypointValid( const std::string& aname, float athreash ) {
	return isKeypointValid( getKeypointIndex(aname), athreash );
}

//---------------------------------------
bool ofxRtmPose::isKeypointValid( int aindex, float athresh ) {
	if (aindex < 0 || aindex >= (int)keypoints.size()) return false;
	const auto& p = keypoints[aindex];
	if( !isKeypointValid( p )) { return false; }
	if (aindex >= (int)scores.size()) return false;
	if (scores[aindex] < athresh) return false;
	return true;
}

//---------------------------------------
bool ofxRtmPose::isKeypointValid( const glm::vec2& akeypoint ) {
	return sIsKeypointValid(akeypoint);
}

//---------------------------------------
bool ofxRtmPose::sIsKeypointValid( const glm::vec2& akeypoint ) {
	if (!std::isfinite(akeypoint.x) || !std::isfinite(akeypoint.y)) return false;
	if (glm::isnan(akeypoint.x) || glm::isnan(akeypoint.y)) return false;
	return true;
}

//---------------------------------------
void ofxRtmPose::drawWithLinkColors(float athresh) {
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
					linesMesh.addColor(linkG.color);
					linesMesh.addColor(linkG.color);
				}
			}
		}
		
		linesMesh.draw();
	} else {
		ofLogWarning("ofxRtmPose") << "drawWithLinkColors no skel spec ptr! | " << ofGetFrameNum();
	}
}

//---------------------------------------
void ofxRtmPose::draw(float athresh) {
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
		
	}
}

//---------------------------------------
void ofxRtmPose::updateFromPose( const ofxRtmPose& apose ) {
	keypoints = apose.keypoints;
	scores = apose.scores;
	score = apose.score;
	box = apose.box;
	
	// now figure out if we have hands //
	configSpec();
	
	_updateHands();
}

//---------------------------------------
void ofxRtmPose::updateFromPoseWithSmoothing( const ofxRtmPose& apose, float aSmoothing ) {
	scores = apose.scores;
	score = apose.score;
	box = apose.box;
	
	size_t numKps = apose.keypoints.size();
	if( numKps != keypoints.size() ) {
		keypoints = apose.keypoints;
		configSpec();
		_updateHands();
		return;
	}
	
	for( std::size_t i = 0; i < numKps; i++ ){
		auto & kp = keypoints[i];
		auto & kpNew = apose.keypoints[i];
		
		bool kpValid = isKeypointValid(kp);
		bool kpNewValid = isKeypointValid(kpNew);
		
		if( !kpValid && kpNewValid ) {
			kp = kpNew;
		}
		if( !kpValid && !kpNewValid ) {
			kp = box.outRect.getCenter();
		} else {
			kp = glm::mix(kp, kpNew, 1.0f-aSmoothing );
		}
	}
	
	configSpec();
	
	_updateHands();
}

//---------------------------------------
void ofxRtmPose::configSpec() {
	mPoseType = PoseType::WHOLE_BODY_133;
	if( keypoints.size() == 21 ) {
		mPoseType = PoseType::HAND_2D_21;
	} else if( keypoints.size() == 133 ) {
		mPoseType = PoseType::WHOLE_BODY_133;
	} else if(keypoints.size() == 26 ) {
		mPoseType = PoseType::BODY_26;
	} else if( keypoints.size() == 17 ) {
		mPoseType = PoseType::BODY_17;
	} else {
		ofLogWarning("RtmSkeleton::configSpec") << "unable to determine pose type from num keypoints: " << keypoints.size() << ", using whole body 133";
	}
	
	auto pname = sGetPoseTypeName(mPoseType);
	if( !ofxRtmSkeletonSpecRegistry::sHasSkeletonSpec(pname) ) {
		ofxRtmSkeletonSpec tspec;
		if( mPoseType == PoseType::WHOLE_BODY_133 ) {
			ofxRtmSkeletonSpecRegistry::sRegisterSkeletonSpec(pname, tspec.getWholebody133Spec() );
		} else if( mPoseType == PoseType::HAND_2D_21 ) {
			ofxRtmSkeletonSpecRegistry::sRegisterSkeletonSpec(pname, tspec.getHand21Spec() );
		} else if( mPoseType == PoseType::BODY_26 ) {
			ofxRtmSkeletonSpecRegistry::sRegisterSkeletonSpec(pname, tspec.getHalpe26Spec() );
		} else if(mPoseType == PoseType::BODY_17 ) {
			ofxRtmSkeletonSpecRegistry::sRegisterSkeletonSpec(pname, tspec.getCoco17() );
		}
	}
	
	mSkelSpecPtr = ofxRtmSkeletonSpecRegistry::sGetSkeletonSpec(pname);
	
}

//---------------------------------------
std::vector<int> ofxRtmPose::getLeftHandIndices() {
	std::vector<int> rindices;
	if( !supportsHands() ) {
		ofLogWarning("ofxRtmPose") << "getLeftHandIndices() : pose type " << sGetPoseTypeName(mPoseType) << " does not have hand information.";
		return rindices;
	}
	
	// Whole Body //
//	// 91–111 left hand (21)
//	N[91]="left_wrist"; N[92]="left_thumb1"; N[93]="left_thumb2"; N[94]="left_thumb3"; N[95]="left_thumb4";
//	N[96]="left_index1"; N[97]="left_index2"; N[98]="left_index3"; N[99]="left_index4";
//	N[100]="left_middle1"; N[101]="left_middle2"; N[102]="left_middle3"; N[103]="left_middle4";
//	N[104]="left_ring1"; N[105]="left_ring2"; N[106]="left_ring3"; N[107]="left_ring4";
//	N[108]="left_pinky1"; N[109]="left_pinky2"; N[110]="left_pinky3"; N[111]="left_pinky4";
	rindices.reserve(21);
	
	if( getPoseType() == HAND_2D_21 ) {
		for( int i = 0; i < 21; i++ ) {
			rindices.push_back(i);
		}
	} else {
		for( int i = 91; i <= 111; i++ ) {
			rindices.push_back(i);
		}
	}
	return rindices;
}

//---------------------------------------
std::vector<int> ofxRtmPose::getRightHandIndices() {
	std::vector<int> rindices;
	if( !supportsHands() ) {
		ofLogWarning("ofxRtmPose") << "getRightHandIndices() : pose type " << sGetPoseTypeName(mPoseType) << " does not have hand information.";
		return rindices;
	}
	
	// Whole Body //
	// 112–132 right hand (21)
	//	N[112]="right_wrist"; N[113]="right_thumb1"; N[114]="right_thumb2"; N[115]="right_thumb3"; N[116]="right_thumb4";
	//	N[117]="right_index1"; N[118]="right_index2"; N[119]="right_index3"; N[120]="right_index4";
	//	N[121]="right_middle1"; N[122]="right_middle2"; N[123]="right_middle3"; N[124]="right_middle4";
	//	N[125]="right_ring1"; N[126]="right_ring2"; N[127]="right_ring3"; N[128]="right_ring4";
	//	N[129]="right_pinky1"; N[130]="right_pinky2"; N[131]="right_pinky3"; N[132]="right_pinky4";
	
	rindices.reserve(21);
	
	if( getPoseType() == HAND_2D_21 ) {
		for( int i = 0; i < 21; i++ ) {
			rindices.push_back(i);
		}
	} else {	
		for( int i = 112; i <= 132; i++ ) {
			rindices.push_back(i);
		}
	}
	return rindices;
}

//---------------------------------------
bool ofxRtmPose::hasValidHand( bool abLeft, float athresh ) {
	if( !mSkelSpecPtr ) {
		return false;
	}
	if( !supportsHands() ) {
//		ofLogWarning("ofxRtmPose") << "pose type " << sGetPoseTypeName(mPoseType) << " does not have hand information.";
		return false;
	}
	
	std::vector<int> indicesToCheck;
	if( abLeft ) {
		indicesToCheck = getLeftHandIndices();
	} else {
		indicesToCheck = getRightHandIndices();
	}
	
	if( indicesToCheck.size() < 1 ) {
		ofLogWarning("ofxRtmPose") << "pose type " << sGetPoseTypeName(mPoseType) << " does not have hand indices.";
		return false;
	}
	
	int numValid = 0;
	for( auto& index : indicesToCheck ) {
		if( isKeypointValid(index,athresh)) {
			numValid++;
		}
	}
	
	// we need most to be valid //
	float pctValid = (float)numValid / ((float)indicesToCheck.size()-1.f);
	if( pctValid > 0.9f ) {
		return true;
	}
	return false;
}



//---------------------------------------
std::vector<glm::vec2> ofxRtmPose::getHandKeypoints( bool abLeft ) {
	if( !supportsHands() ) {
		ofLogWarning("ofxRtmPose") << "pose getHandKeypoints " << sGetPoseTypeName(mPoseType) << " does not have hand information.";
		return std::vector<glm::vec2>();
	}
	if( keypoints.size() < 10 ) {
		return std::vector<glm::vec2>();
	}
	
	auto indices = getLeftHandIndices();
	if( !abLeft ) {
		indices = getRightHandIndices();
	}
	
	return sliceVectorInclusive( keypoints, indices );
}

//---------------------------------------
std::vector<float> ofxRtmPose::getHandScores( bool abLeft ) {
	if( !supportsHands() ) {
		ofLogWarning("ofxRtmPose") << "pose getHandScores " << sGetPoseTypeName(mPoseType) << " does not have hand information.";
		return std::vector<float>();
	}
	if( keypoints.size() < 10 ) {
		return std::vector<float>();
	}
	
	auto indices = getLeftHandIndices();
	if( !abLeft ) {
		indices = getRightHandIndices();
	}
	
	return sliceVectorInclusive( scores, indices );
}

//---------------------------------------
void ofxRtmPose::_updateHands() {
	
	auto clearHand = [](ofxRtmHand& ahand) {
		ahand.keypoints.clear();
		ahand.scores.clear();
		ahand.score = 0.f;
	};
	
	if(!supportsHands()) {
		clearHand(mLeftHand);
		clearHand(mRightHand);
		return;
	}
	
	
	auto computeAvgScore = [](std::vector<float>& ascores ) -> float {
		float ts = 0.f;
		float totalSF = ascores.size();
		if( totalSF < 1.f ) { return ts; }
		for( auto& sc : ascores ) {
			ts += sc;
		}
		return ts / totalSF;;
	};
	
	mLeftHand.keypoints = getHandKeypoints(true);
	mLeftHand.scores = getHandScores(true);
	mLeftHand.score = computeAvgScore(mLeftHand.scores);
	mLeftHand.handed = ofxRtmHand::LEFT;
	
	mRightHand.keypoints = getHandKeypoints(false);
	mRightHand.scores = getHandScores(false);
	mRightHand.score = computeAvgScore(mRightHand.scores);
	mRightHand.handed = ofxRtmHand::RIGHT;
}
