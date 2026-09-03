#include "ofApp.h"
#include "ofxRtmModelLoader.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetVerticalSync(false);
	
	
	
	auto pfunc = [this](const ofHttpRequest & req, float progress) {
		ofLogNotice("ofApp::setup") << "Download model progress: " << progress;
	};
	
	
	ofxRtmPoseTracker::Settings rsettings;
	
	ofHttpRequest req;
	// uncomment if using a newer OF version to get progress output.
	//req.progressCallback = pfunc;
	req.timeoutSeconds = 60 * 15; // 15 minutes
	
	ofLogNotice("ofApp") << "Downloading models.";
	// download the model you would like to use.
	// There are model urls listed at the top of the ofxRtmModelLoader.h file for convenience.
	
	//	std::string loadModel(const std::string& aTag, const std::string& modelsRoot, const std::string& modelUrl );
	bool bUseRtmo = false;
	
	if( bUseRtmo ) {
		
		rsettings.poseModelPath = ofxRtmModelLoader::loadModel("rtmo",
															   "models/Rtmo",
															   "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
															   req);
		
	} else {
		
		
		rsettings.detectorModelPath = ofxRtmModelLoader::loadModel("yolox_m",
																   "models/RtmDetectors",
																   "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
																   req);
		
		rsettings.poseModelPath = ofxRtmModelLoader::loadModel("rtmpose-l-384x288",
															   "models/RtmPoseWholeBody",
															   "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip",
															   req);
	}
	mRtmPoseTracker = make_shared<ofxRtmPoseTracker>();
	mRtmPoseTracker->setup(rsettings);
	mRtmPoseTracker->start(); // start a thread
	
	gui.setup(mRtmPoseTracker->getParams("RtmParams"));
	
	gui.loadFromFile("settings.xml");
	
	
}

//--------------------------------------------------------------
void ofApp::update(){
	mBNewFrame = false;
	
		
	if( !mGrabber.isInitialized() ) {
		mGrabber.listDevices();
		mGrabber.setup(1280, 720);
	}
		
	if( mGrabber.getWidth() > 1 ) {
		mGrabber.update();
		if( mGrabber.isFrameNew()) {
			mBNewFrame = true;
			mPixels = mGrabber.getPixels();
			mPixels.mirror(false, true);
		}
	}
	
	
	if(mBNewFrame) {
		if( mPixels.isAllocated() && mPixels.getWidth() > 1 ) {
			
			ofRectangle wrect( 0, 0, ofGetWidth(), ofGetHeight() );
			ofRectangle frect( 0, 0, mPixels.getWidth(), mPixels.getHeight() );
			frect.scaleTo(wrect, OF_SCALEMODE_FIT );
			
			if( mRtmPoseTracker ) {
				mRtmPoseTracker->setOutRect(frect);
				mRtmPoseTracker->run(mPixels);
			}
			if( mVidTex.isAllocated() && (mVidTex.getWidth() != mPixels.getWidth() || mVidTex.getHeight() != mPixels.getHeight()) ) {
				ofLogNotice("ofApp::update") << "clearing video texture.";
				mVidTex.clear();
			}
			mVidTex.loadData(mPixels);
		}
	}
}

//--------------------------------------------------------------
void ofApp::draw(){
	if( mVidTex.getWidth() > 0 && mVidTex.isAllocated() ) {
		ofSetColor(255);
		if(mRtmPoseTracker) {
			mVidTex.draw(mRtmPoseTracker->getOutRect());
			mRtmPoseTracker->draw();
		}
	}
	
	std::stringstream ss;
	ss << "app fps: " << int(ofGetFrameRate());
	ss << endl << "tracker fps: " << int(mRtmPoseTracker->getFps());
	
	ofDrawBitmapStringHighlight(ss.str(), ofGetWidth()-220, 20 );
	
	gui.draw();
}

//--------------------------------------------------------------
void ofApp::exit(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseScrolled(int x, int y, float scrollX, float scrollY){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
