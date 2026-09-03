

#include "ofxRtmPoseTracker.h"
#include "ofFileUtils.h"

using namespace std;

//--------------------------------------------------------------
ofxRtmPoseTracker::ofxRtmPoseTracker() {
	mBoxThresh.set("BoxThresh", 0.5f, 0.f, 1.f);
	mMinKeypointScoreToTrack.set("MinKeypointScoreToTrack", 0.5f, 0.0f, 4.f);
	mMinPoseSize.set("MinPoseSize", 0, 0, 512);
	mMinHandSize.set("MinHandSize", 0, 0, 512);
	mDurNotFoundToKill.set("DurNotFoundToKill", 1.f, 0.0f, 3.f);
	
	mBRemoveOverlappingPoses.set("RemoveOverlapPoses", false);
	mBRemoveOverlappingHands.set("RemoveOverlappingHands", true);
	mOverlapThreshN.set("OverlapThreshN", 0.5f, 0.0f, 1.f);
	mHandOverlapThresh.set("HandOverlapThresh", 0.5f, 0.0f, 1.f);
	
	mBDrawPoses.set("DrawPoses", true );
	mBDrawHands.set("DrawHands", true );
	mDrawThresh.set("DrawThresh", 0.5f, 0.f, 1.f);
	mPosSmoothing.set("PosSmoothPct", 0.f, 0.f, 1.f);
	mBDrawBoxes.set("DrawBoxes", true);
	mBDrawJoints.set("DrawJoints", true);
	mBColorById.set("ColorById", true);
	mBDrawSrcPoses.set("DrawSrcPoses", false );
	
	mPoseMatchCostThresh.set("PoseMatchCostThresh", 1.0f, 0.0f, 10.0f);
	mHandMatchCostThresh.set("HandMatchCostThresh", 1.0f, 0.0f, 3.f);
}

//--------------------------------------------------------------
ofxRtmPoseTracker::~ofxRtmPoseTracker() {
	_removeAppListeners();
	stop();
	mBHasNewData = false;
}

//--------------------------------------------------------------
ofParameterGroup& ofxRtmPoseTracker::getParams( std::string aname ) {
	if( mParams.size() < 1 ) {
		mParams.setName(aname);
		mParams.add( mBoxThresh );
		mParams.add( mMinKeypointScoreToTrack );
		mParams.add( mMinPoseSize );
		mParams.add( mMinHandSize );
		mParams.add( mPoseMatchCostThresh );
		mParams.add( mHandMatchCostThresh );
		mParams.add( mDurNotFoundToKill );
		
		mParams.add( mBRemoveOverlappingPoses );
		mParams.add( mOverlapThreshN );
		
		mParams.add( mBRemoveOverlappingHands );
		mParams.add(mHandOverlapThresh);
		
		
		mParams.add( mBDrawPoses );
		mParams.add( mBDrawHands );
		mParams.add( mDrawThresh );
		mParams.add( mPosSmoothing );
		mParams.add( mBColorById );
		mParams.add( mBDrawBoxes );
		mParams.add( mBDrawJoints );
		mParams.add( mBDrawSrcPoses );
	}
	return mParams;
}

//--------------------------------------------------------------
bool ofxRtmPoseTracker::setup( ofxRtmPoseTracker::Settings asettings ) {
	
	if( asettings.detectorModelPath.empty() && asettings.poseModelPath.empty() ) {
		ofLogError("ofxRtmPoseTracker") << "Must pass in a valid detectorModelPath || poseModelPath to an onnx model.";
		return false;
	}
	
	if(mRtmPoseDetector) {
		mRtmPoseDetector.reset();
	}
	mRtmPoseDetector = make_shared<ofxRtmPoseDetector>();
	mBLoaded = false;
	if( asettings.detectorModelPath.empty() ) {
		mBLoaded = mRtmPoseDetector->load(asettings.poseModelPath, asettings.logLevel);
	} else {
		mBLoaded = mRtmPoseDetector->load(asettings.detectorModelPath, asettings.poseModelPath, asettings.logLevel );
	}
	
	if( !mBLoaded ) {
		ofLogError("ofxRtmPoseTracker") << "unable to load from: " << asettings.detectorModelPath << " pose: " << asettings.poseModelPath;
		mRtmPoseDetector.reset();
	}
	return mBLoaded;
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::clear() {
	if( !isThreadRunning() ) {
		mPoses.clear();
		mHands.clear();
		mInPoses.clear();
		mBHasNewData = false;
	} else {
		ofLogNotice("ofxRtmPoseTracker::clear()") << "clear not implemented while thread running!";
	}
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::start() {
	if (isThreadRunning()) return;
	startThread();
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::stop() {
	if( isThreadRunning() ) {
		{
			// wake thread if it's sleeping
			std::lock_guard<std::mutex> lk(mCvMutex);
			mBWakeThread = true;
		}
		mCondition.notify_one();
		waitForThread(true, 3000);
	}
	mBHasNewData = false;
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::update( ofEventArgs& args ) {
	mBHasNewData = false;
	
	float dt = ofClamp( ofGetLastFrameTime(), 1.f / 9999.f, 1.f / 5.f );
	
	// check about the poses //
	if( isThreadRunning() ) {
		bool bNewTPoses = mBHasNewThreadedPoses.load();
		if( bNewTPoses ) {
			
			{
				std::unique_lock<std::mutex> lck(mCopyMutex);
				mInPoses = mThreadedPoses;
				mBHasNewThreadedPoses = false;
			}
			
			mCondition.notify_one();
			
			_matchPoses(mInPoses, mPoses);
			_matchHands(mInPoses, mHands );
			_assocTrackedHandsWithTrackedPoses(mPoses, mHands);
			mFpsCounter.newFrame();
			
			mBHasNewData = true;
			
		}
	}
	
	
	for( auto& pose : mPoses ) {
		pose->age += dt;
		pose->timeNotFound += dt;
	}
	
	for( auto& hand : mHands ) {
		hand->age += dt;
		hand->timeNotFound += dt;
	}
	
	if( mPoses.size() < 1 ) {
		if( mIdCounter > 999999 ) {
			// just in case //
			mIdCounter = 0;
		}
	}
	
	if( mHands.size() < 1 ) {
		if( mIdCounterHand > 999999 ) {
			// just in case //
			mIdCounterHand = 0;
		}
	}
	
}

//--------------------------------------------------------------
bool ofxRtmPoseTracker::hasNewData() {
	if(!mRtmPoseDetector) {
		return false;
	}
	return mBHasNewData;
}

//--------------------------------------------------------------
bool ofxRtmPoseTracker::run(ofPixels& apixels) {
	if(!mRtmPoseDetector) {
		return false;
	}
	
	if(mBEnableUpdateListeners) {
		_addAppListeners();
	} else {
		_removeAppListeners();
	}
	
	if( isThreadRunning() ) {
		bool bLockSucces = false;
		
		
		if( tryLock()) {
			if( !mBHasPendingPixels ) {
				mPixelsPending = apixels;      // deep copy
				mBHasPendingPixels = true;
				
				mRtmPoseDetector->setBoxThresh(mBoxThresh);
				mRtmPoseDetector->setDrawThresh(mDrawThresh);
				mRtmPoseDetector->setRemoveOverlappingPoses(mBRemoveOverlappingPoses);
				mRtmPoseDetector->setOverlapToRemoveThresh(mOverlapThreshN);
				mRtmPoseDetector->setAddDetectBoxIfNonDetected(true); 
				bLockSucces = true;
			}
			unlock();
		}
		
		
		if(bLockSucces) {
			std::lock_guard<std::mutex> lk(mCvMutex);
			mBWakeThread = true;
		}
		mCondition.notify_one();
	} else {
		
		mRtmPoseDetector->setBoxThresh(mBoxThresh);
		mRtmPoseDetector->setDrawThresh(mDrawThresh);
		mRtmPoseDetector->setRemoveOverlappingPoses(mBRemoveOverlappingPoses);
		mRtmPoseDetector->setOverlapToRemoveThresh(mOverlapThreshN);
		mRtmPoseDetector->setAddDetectBoxIfNonDetected(true); 

		
		bool bRunOk = mRtmPoseDetector->run(apixels);
		mInPoses = mRtmPoseDetector->getPoses();
		_matchPoses( mInPoses, mPoses );
		_matchHands( mInPoses, mHands );
		_assocTrackedHandsWithTrackedPoses(mPoses, mHands);
		
		mBHasNewData = bRunOk;
		
		mFpsCounter.newFrame();
		return bRunOk;
	}
	
	return true;
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::draw() {
	if(mRtmPoseDetector) {
		auto orect = mRtmPoseDetector->getOutRect();
		draw( orect.x, orect.y, orect.width, orect.height );
	}
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::draw( const ofRectangle& arect ) {
	draw( arect.x, arect.y, arect.width, arect.height );
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::draw(float ax, float ay, float awidth, float aheight) {
	if(!mRtmPoseDetector) return;
	
	const auto outRect = mRtmPoseDetector->getOutRect();
	float scaleX = awidth / outRect.width;
	float scaleY = aheight / outRect.height;
	
	std::vector<ofColor> idColors = {ofColor::cyan, ofColor::magenta, ofColor::orange, ofColor::purple, ofColor::red };
	
	ofPushMatrix(); {
		ofTranslate(ax, ay);
		ofScale(scaleX, scaleY);
		
		ofTranslate(-outRect.x, -outRect.y);
		
		if( mBDrawSrcPoses && mRtmPoseDetector ) {
//			auto& srcPoses = mRtmPoseDetector->getPoses();
			for( auto& srcPose : mInPoses ) {
				//ofLogNotice("ofxRtmPoseTracker") << "srcPose type: " << ofxRtmPose::sGetPoseTypeName(srcPose.getPoseType());
				srcPose.draw(mDrawThresh);
				// for( auto& kp : srcPose.keypoints ) {
				// 	ofDrawLine(glm::vec2(20, 20), kp);
				// }
			}
		}
		
		if( mBDrawPoses ) {
			for( auto& pose : mPoses ) {
				if(mBColorById) {
					ofSetColor( idColors[pose->ID % idColors.size() ] );
					pose->draw(mDrawThresh);
					ofDrawBitmapStringHighlight("ID: " +ofToString(pose->ID, 0), pose->box.outRect.getCenter() );
				} else {
					pose->drawWithLinkColors( mDrawThresh );
				}
				
				if( mBDrawHands ) {
//					for( auto& hand : pose->hands ) {
//						ofDrawLine( pose->keypoints[1], hand->keypoints[0] );
//					}
				}
				
				if( mBDrawJoints ) {
					auto& kps = pose->keypoints;
					for( size_t j = 0; j < kps.size(); j++ ) {
						if( pose->isKeypointValid(j, mDrawThresh )) {
							ofSetColor( idColors[pose->ID % idColors.size() ] );
							ofDrawCircle(kps[j], 2.f);
//							if( j < pose->velocities.size() ) {
//								//							auto vel = pose->keypoints[j] - pose->prevKeypoints[j];
//								ofSetColor( idColors[(pose->ID+2) % idColors.size() ] );
//								ofDrawLine( kps[j], kps[j] + pose->velocities[j] );
//							}
						}
					}
				}
				if( mBDrawBoxes ) {
					ofSetColor( ofColor::yellow );
					ofNoFill();
					ofDrawRectangle(pose->box.outRect);
					ofFill();
				}
			}
		}
		if( mBDrawHands ) {
						
			ofNoFill();
			for( auto& hand : mHands ) {
				ofSetColor( idColors[ (hand->ID+1) % idColors.size() ] );
				hand->draw(mDrawThresh);
				
				if( mBDrawJoints ) {
					auto& kps = hand->keypoints;
					for( size_t j = 0; j < kps.size(); j++ ) {
						ofSetColor( idColors[ (hand->ID+2) % idColors.size() ] );
						ofDrawCircle(kps[j], 10.f);
					}
				}
				ofDrawBitmapStringHighlight("id: "+ofToString(hand->ID, 0), hand->getPalmPos() );
			}
			ofFill();
		}
	} ofPopMatrix();
	
	if( mRtmPoseDetector && mBDrawSrcPoses) {
		ofDrawBitmapStringHighlight("Num Raw poses: " +ofToString(mRtmPoseDetector->getPoses().size(), 0), ax+awidth-180, ay+24 );
	}
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::drawHand( std::shared_ptr<ofxRtmTrackedHand> ahand, const ofRectangle& arect ) {
	if(!mRtmPoseDetector) return;
	if(!ahand) return;
	
	const auto outRect = mRtmPoseDetector->getOutRect();
	float scaleX = arect.width / outRect.width;
	float scaleY = arect.height / outRect.height;
	
	std::vector<ofColor> idColors = {ofColor::cyan, ofColor::magenta, ofColor::orange, ofColor::purple, ofColor::red };
	
	ofPushMatrix(); {
		ofTranslate(arect.x, arect.y);
		ofScale(scaleX, scaleY);
		
		ofTranslate(-outRect.x, -outRect.y);
		
		ofNoFill();
		ofSetColor( idColors[ (ahand->ID+1) % idColors.size() ] );
		ahand->draw(mDrawThresh);
		
		if( mBDrawJoints ) {
			auto& kps = ahand->keypoints;
			for( size_t j = 0; j < kps.size(); j++ ) {
				ofSetColor( idColors[ (ahand->ID+2) % idColors.size() ] );
				ofDrawCircle(kps[j], 10.f);
			}
		}
		ofFill();
		
	} ofPopMatrix();
	
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::threadedFunction() {
	ofPixels currentPix;
	
	while (isThreadRunning()) {
		
		// 1) Wait until there is pending work (or we’re stopping)
		if (!hasPendingPixels()) {
			std::unique_lock<std::mutex> lk(mCvMutex);
			mCondition.wait(lk, [&]{
				return !isThreadRunning() || mBWakeThread;
			});
			mBWakeThread = false;
			
			if (!isThreadRunning()) break;
			if (!hasPendingPixels()) continue;
		}
		
		// 2) Grab the latest pending frame (atomic-ish swap under lock)
		
		lock();
		if( mBHasPendingPixels ) {
//			ofLogNotice("ofxRtmPoseTracker::threadedFunction") << "copying pixels.";
			currentPix = mPixelsPending;
			mBHasPendingPixels = false;
//			bYeahNewPix = true;
		}
		unlock();
		
		// 3) Process
		if(mRtmPoseDetector) {
//			ofLogNotice("ofxRtmPoseTracker::threadedFunction") << "running pose detector.";
			bool bRunOk = mRtmPoseDetector->run(currentPix);
			
			if(bRunOk){
				std::unique_lock<std::mutex> lck(mCopyMutex);
//				ofLogNotice("ofxRtmPoseTracker::threadedFunction") << "copying poses.";
				mThreadedPoses = mRtmPoseDetector->getPoses();
				mBHasNewThreadedPoses = true;
			}
		}
			
			
		
		// 4) Loop again immediately; if new frames arrived during processing,
		//    hasPending() will be true and we’ll take them without sleeping.
	}
}




//--------------------------------------------------------------
float ofxRtmPoseTracker::_poseMatchCost(ofxRtmPose& det, shared_ptr<ofxRtmTrackedPose>& tracked, const ofRectangle& aFrameRect) {
	
	ofRectangle detBox = det.getBoundingBox();
	ofRectangle trackedBox = tracked->getBoundingBox();
	
	float dist = (fabs(detBox.getWidth()-trackedBox.getWidth()) / aFrameRect.width) + (fabs(detBox.getHeight()-trackedBox.getHeight() ) / aFrameRect.height);
	
	float numPtsChecked = 0.f;
	float ppscore = 99999.f;
	if( det.keypoints.size() == tracked->keypoints.size() ) {
		auto numKps = det.keypoints.size();
		float totalDist2 = 0.f;
		for( int i = 0; i < numKps; i++ ) {
			if( det.isKeypointValid(i, 0.1f) && tracked->isKeypointValid(i, 0.1f)) {
				totalDist2 += glm::distance2(det.keypoints[i], tracked->keypoints[i]);
				numPtsChecked += 1.f;
			}
		}
		
		if( numPtsChecked > 1.f ) {
			float td = glm::sqrt(totalDist2);
			td /= aFrameRect.height; // kinda normalize
			td /= numPtsChecked;
			ppscore = td;
		}
		
	}
	
	// 1 - IoU (so lower = better match)
	float iou = ofxRtmPoseDetector::sComputeIoU(det.getBoundingBox(), tracked->getBoundingBox());
	float iouCost = 1.0f - iou;
	
	// Weighted combo — tune these
	return 0.25f * dist + 0.75f * iouCost + 0.75f * ppscore;
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::_matchPoses( std::vector<ofxRtmPose>& aIncomingPoses, std::vector<std::shared_ptr<ofxRtmTrackedPose>>& aTrackedPoses ) {
//	float frameRate = ofClamp( mFpsCounter.getFps(), 1, 99999 );

//	int mNumFramesToDie = mDurNotFoundToKill * frameRate;
//	if( mNumFramesToDie < 1 ) {mNumFramesToDie =1;}
	
	auto outRect = getOutRect();
	
	for( auto& pose : aTrackedPoses ) {
		pose->bFoundThisFrame = false;
		pose->prevKeypoints = pose->keypoints;
		
	}
	
	// TODO: filtering of poses.
	float minKeypointScoreToTrack = mMinKeypointScoreToTrack;
	std::vector<ofxRtmPose> inPoses;
	for( auto& apose : aIncomingPoses ) {
		auto bbox = apose.getBoundingBox();
		float tMaxSize = std::max( bbox.getWidth(), bbox.getHeight() );
		if( tMaxSize < mMinPoseSize ) {
			continue;
		}
		
		// loop through all of the keypoints
		float numKpsValid = 0.f;
		for( int i = 0; i < apose.keypoints.size(); i++ ) {
			if( apose.isKeypointValid( i, minKeypointScoreToTrack )) {
				numKpsValid += 1.f;
			}
		}
		if( numKpsValid < 3 ) {continue;}
		float numKps = apose.keypoints.size()-1.f;
		if( numKps < 3 ) {continue;}
		float pctValid = numKpsValid / numKps;
		if( pctValid < 0.8f ) {
			continue;
		}
		inPoses.push_back(apose);
	}
	
	// Build and solve:
	std::vector<std::vector<float>> cost(inPoses.size(), std::vector<float>(aTrackedPoses.size()));
	for (int i = 0; i < inPoses.size(); i++) {
		for (int j = 0; j < aTrackedPoses.size(); j++) {
			cost[i][j] = _poseMatchCost(inPoses[i], aTrackedPoses[j], outRect);
		}
	}
		
	auto assignment = _hungarianMatch(cost, /*maxCostThreshold=*/ mPoseMatchCostThresh);
	
	std::vector<bool> trackedMatched(aTrackedPoses.size(), false);
	
	for (int i = 0; i < inPoses.size(); i++) {
		shared_ptr<ofxRtmTrackedPose> myPose;
		
		if (assignment[i] >= 0) {
			myPose = aTrackedPoses[assignment[i]];
			// Update existing tracked pose
			trackedMatched[assignment[i]] = true;
		} else {
			// No match — spawn new tracked pose
			myPose = std::make_shared<ofxRtmTrackedPose>(inPoses[i]);
			myPose->ID = (mIdCounter++);
			myPose->prevKeypoints = myPose->keypoints;
			aTrackedPoses.push_back(myPose);
		}
		
		if( myPose ) {
			myPose->timeNotFound = 0.f;
			// disable search for next incoming pose
			myPose->bFoundThisFrame = true;
			
			if( mPosSmoothing < 0.001f || mPosSmoothing > 0.99f ){
				myPose->updateFromPose( inPoses[i] );
			} else {
				myPose->updateFromPoseWithSmoothing( inPoses[i], mPosSmoothing);
			}
		}
	}
	
	for( auto& pose : aTrackedPoses ) {
		if(!pose->bFoundThisFrame) {
			if( pose->timeNotFound > mDurNotFoundToKill ) {
				pose->bRemove = true;
			}
		} else {
			pose->timeNotFound = 0.f;
		}
	}
	
	ofRemove( aTrackedPoses, [](const shared_ptr<ofxRtmTrackedPose>& apose) {
		return apose->bRemove;
	});
}

//--------------------------------------------------------------
float ofxRtmPoseTracker::_handMatchCost(const ofxRtmHand& det, const shared_ptr<ofxRtmTrackedHand>& tracked, const ofRectangle& aFrameRect) {

	// auto palmDiff = det.getPalmPos() - tracked->getPalmPos();
	auto palmDiff = (det.getPalmPos() - tracked->getPalmPos()) / glm::vec2(aFrameRect.width, aFrameRect.height);
	float dist = glm::length(palmDiff);

	float numPtsChecked = 0.f;
	float ppscore = 99999.f;
	if( det.keypoints.size() == tracked->keypoints.size() ) {
		auto numKps = det.keypoints.size();
		float totalDist2 = 0.f;
		for( int i = 0; i < numKps; i++ ) {
			if( det.isKeypointValid(i, 0.1f) && tracked->isKeypointValid(i, 0.1f)) {
				totalDist2 += glm::distance2(det.keypoints[i], tracked->keypoints[i]);
				numPtsChecked += 1.f;
			}
		}

		if( numPtsChecked > 1.f ) {
			float td = glm::sqrt(totalDist2);
			td /= aFrameRect.height; // kinda normalize
			td /= numPtsChecked;
			ppscore = td;
		}

	}

    // 1 - IoU (so lower = better match)
    float iou = ofxRtmPoseDetector::sComputeIoU(det.getBoundingBox(), tracked->getBoundingBox());
    float iouCost = 1.0f - iou;

    // Weighted combo — tune these
    return 0.25f * dist + 0.75f * iouCost + 0.75f * ppscore;
}

//--------------------------------------------------------------
float ofxRtmPoseTracker::_handPoseMatchCost(const shared_ptr<ofxRtmTrackedPose> apose,
											const shared_ptr<ofxRtmTrackedHand>& ahand, const ofRectangle& aFrameRect) {
	
	// we have two hands in the pose, so perform match on both and then use the lower score
	auto& lhand = apose->getLeftHand();
	float lscore = _handMatchCost(lhand, ahand, aFrameRect );
	
	auto& rhand = apose->getRightHand();
	float rscore = _handMatchCost(rhand, ahand, aFrameRect );
	
	return std::min(lscore, rscore);
	
}


// Cost matrix: rows = incoming detections, cols = tracked hands
// Returns assignment: assignment[i] = j means incoming[i] -> tracked[j], or -1 if unmatched
//--------------------------------------------------------------
std::vector<int> ofxRtmPoseTracker::_hungarianMatch( const std::vector<std::vector<float>>& costMatrix, float maxCostThreshold) {
	// if (costMatrix.empty() || costMatrix[0].empty()) return {};
	if (costMatrix.empty()) return {};

    int n = costMatrix.size();      // incoming detections
    int m = costMatrix[0].size();   // tracked hands
    int sz = std::max(n, m);

    // Pad to square with high cost
    std::vector<std::vector<float>> C(sz, std::vector<float>(sz, 1e9f));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i][j] = costMatrix[i][j];

    std::vector<float> u(sz + 1), v(sz + 1);
    std::vector<int>   p(sz + 1, 0), way(sz + 1, 0);

    for (int i = 1; i <= sz; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minVal(sz + 1, 1e18f);
        std::vector<bool>  used(sz + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = -1;
            float delta = 1e18f;
            for (int j = 1; j <= sz; j++) {
                if (!used[j]) {
                    float cur = C[i0-1][j-1] - u[i0] - v[j];
                    if (cur < minVal[j]) { minVal[j] = cur; way[j] = j0; }
                    if (minVal[j] < delta) { delta = minVal[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= sz; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minVal[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }

    // Extract assignments for real incoming detections only
    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= m; j++) {
        int i = p[j] - 1; // 0-indexed incoming detection
        if (i >= 0 && i < n) {
            float cost = costMatrix[i][j-1];
            if (cost < maxCostThreshold)
                assignment[i] = j - 1; // matched to tracked hand j-1
        }
    }
    return assignment;
}

//--------------------------------------------------------------
std::vector<ofxRtmHand> ofxRtmPoseTracker::_suppressOverlappingHands( const std::vector<ofxRtmHand>& incoming, float iouThreshold) {
    // Sort by confidence descending — keep the more confident detection
    std::vector<ofxRtmHand> sorted = incoming;
    std::sort(sorted.begin(), sorted.end(), [](const ofxRtmHand& a, const ofxRtmHand& b) {
        return a.getScore() > b.getScore();
    });

    std::vector<bool> suppressed(sorted.size(), false);

    for (int i = 0; i < sorted.size(); i++) {
		if (suppressed[i]) {continue;}
        //ofRectangle bboxI = getBoundingBox(sorted[i].keypoints);
		ofRectangle bboxI = sorted[i].getBoundingBox();

        for (int j = i + 1; j < sorted.size(); j++) {
			if (suppressed[j]) {continue;}
            // ofRectangle bboxJ = getBoundingBox(sorted[j].keypoints);
			ofRectangle bboxJ = sorted[j].getBoundingBox();

			if (ofxRtmPoseDetector::sComputeIoU(bboxI, bboxJ) > iouThreshold) {
				suppressed[j] = true;
			}
        }
    }

    std::vector<ofxRtmHand> result;
	for (int i = 0; i < sorted.size(); i++) {
		if (!suppressed[i]) {
			result.push_back(sorted[i]);
		}
	}

    return result;
}


//--------------------------------------------------------------
void ofxRtmPoseTracker::_matchHands( std::vector<ofxRtmPose>& aIncomingPoses, std::vector<std::shared_ptr<ofxRtmTrackedHand>>& aTrackedHands ) {
	
	for( auto& hand : mHands ) {
		hand->bFoundThisFrame = false;
	}
	
	auto outRect = getOutRect();
	
	// extract incoming hands
	std::vector<ofxRtmHand> inHands;
	
	auto isHandValidT = []( ofxRtmHand& ahand, ofRectangle& aOutRect ) -> bool {
		auto& kps = ahand.keypoints;
		if( kps.size() < 2 ) {
			return false;
		}
		float numInBounds = 0.f;
		for( auto& kp : kps ) {
			if( aOutRect.inside(kp.x, kp.y)) {
				numInBounds += 1.f;
			}
		}
		float numInbPct = (numInBounds / ((float)kps.size()-1.f));// > 0.8f;
		if( numInbPct < 0.5f ) return false;

//		bool bgoodLengths = ahand.hasPlausibleFingerLengths(3.);
//		if( !bgoodLengths ) {
			//return false;
//		}

		return true;
	};
	
	float minKeypointScoreToTrack = mMinKeypointScoreToTrack;
	
	// loop through poses and track some hands
	for( auto& npose : aIncomingPoses ) {
//		if( npose.age > 1.0f && npose->timeNotFound < 0.5f ) {
		if( npose.hasValidHand(true, minKeypointScoreToTrack )) {
			auto lhand = npose.getLeftHand();
			
			auto bbox = lhand.getBoundingBox();
			float tMaxSize = std::max( bbox.getWidth(), bbox.getHeight() );
			if( tMaxSize < mMinHandSize ) {
				continue;
			}
			
			if( isHandValidT(lhand, outRect)) {
				inHands.push_back( lhand );
			}
		}
		if( npose.hasValidHand(false, minKeypointScoreToTrack)) {
			auto rhand = npose.getRightHand();
			
			auto bbox = rhand.getBoundingBox();
			float tMaxSize = std::max( bbox.getWidth(), bbox.getHeight() );
			if( tMaxSize < mMinHandSize ) {
				continue;
			}
			
			if( isHandValidT(rhand, outRect)) {
				inHands.push_back(rhand);
			}
		}
//		}
	}
	
	if(mBRemoveOverlappingHands) {
		inHands = _suppressOverlappingHands( inHands, mHandOverlapThresh );
	}

	// Build and solve:
	std::vector<std::vector<float>> cost(inHands.size(), std::vector<float>(mHands.size()));
	for (int i = 0; i < inHands.size(); i++) {
		for (int j = 0; j < mHands.size(); j++) {
			cost[i][j] = _handMatchCost(inHands[i], mHands[j], outRect);
		}
	}

	auto assignment = _hungarianMatch(cost, /*maxCostThreshold=*/ mHandMatchCostThresh);
	
	std::vector<bool> trackedMatched(mHands.size(), false);

	for (int i = 0; i < inHands.size(); i++) {
		shared_ptr<ofxRtmTrackedHand> myHand;

		if (assignment[i] >= 0) {
			myHand = mHands[assignment[i]];
			// Update existing tracked hand
			// mHands[assignment[i]].update(incoming[i]);
			trackedMatched[assignment[i]] = true;
		} else {
			// No match — spawn new tracked hand
			// tracked.push_back(TrackedHand(incoming[i]));
			myHand = std::make_shared<ofxRtmTrackedHand>(inHands[i]);
			myHand->ID = (mIdCounterHand++);
			mHands.push_back(myHand);
		}

		if( myHand ) {
			myHand->bFoundThisFrame = true;
			if( mPosSmoothing < 0.001f || mPosSmoothing > 0.99f ){
				myHand->updateFromHand( inHands[i] );
			} else {
				myHand->updateFromHandWithSmoothing(inHands[i], mPosSmoothing);
			}
		}

	}
	
	for( auto& hand : mHands ) {
		if(!hand->bFoundThisFrame) {
			if( hand->timeNotFound > mDurNotFoundToKill ) {
				hand->bRemove = true;
			}
		} else {
			hand->timeNotFound = 0.f;
		}
	}
	
	ofRemove( mHands, [](const shared_ptr<ofxRtmTrackedHand>& ahand) {
		return ahand->bRemove;
	});
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::_assocTrackedHandsWithTrackedPoses(std::vector<std::shared_ptr<ofxRtmTrackedPose>> aPoses,
										std::vector<std::shared_ptr<ofxRtmTrackedHand>> aHands) {
	
	
	if (aPoses.empty() || aHands.empty()) return;
	
	for( auto& hand : aHands ) {
		hand->poseID = -1;
	}
	
	auto outRect = getOutRect();
	
	// Build cost matrix with 2 slots per pose (rows = pose slots, cols = hands)
	// Pose i occupies rows [i*2] and [i*2+1]
	int numSlots = aPoses.size() * 2;
	int numHands = aHands.size();
	
	std::vector<std::vector<float>> cost(numSlots, std::vector<float>(numHands));
	
	for (int i = 0; i < (int)aPoses.size(); i++) {
		for (int j = 0; j < numHands; j++) {
			float c = _handPoseMatchCost(aPoses[i], aHands[j], outRect);
			cost[i * 2    ][j] = c;  // slot A for pose i
			cost[i * 2 + 1][j] = c;  // slot B for pose i
		}
	}
	
	auto assignment = _hungarianMatch(cost, mHandMatchCostThresh);
	
	// Collect assignments back onto poses
	// Each pose can receive up to 2 hands
	for (int i = 0; i < (int)aPoses.size(); i++) {
		aPoses[i]->hands.clear();
		
		for (int slot = 0; slot < 2; slot++) {
			int handIdx = assignment[i * 2 + slot];
			if (handIdx >= 0) {
				aPoses[i]->hands.push_back(aHands[handIdx]);
				aHands[handIdx]->poseID = aPoses[i]->ID;
			}
		}
	}
}

//--------------------------------------------------------------
void ofxRtmPoseTracker::_addAppListeners() {
	if( !mBHasAppListeners ) {
		ofAddListener(ofEvents().update, this, &ofxRtmPoseTracker::update );
	}
	mBHasAppListeners=true;
}

//------------------------------------------------------------------------
void ofxRtmPoseTracker::_removeAppListeners() {
	if(mBHasAppListeners) {
		ofRemoveListener(ofEvents().update, this, &ofxRtmPoseTracker::update );
	}
	mBHasAppListeners=false;
}
