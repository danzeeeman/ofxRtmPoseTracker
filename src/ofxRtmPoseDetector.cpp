

#include "ofxRtmPoseDetector.h"

const std::vector<float> IMAGE_MEAN{ 123.675, 116.28, 103.53 };
const std::vector<float> IMAGE_STD{ 58.395, 57.12, 57.375 };


// Convert cv::Mat (8-bit) into ofPixels (uint8).
// Returns true on success. Set swapBgrToRgb=true for typical OpenCV BGR->RGB.
inline bool cvMatToOfPixels(const cv::Mat& src,
							ofPixels& dst,
							bool swapBgrToRgb)
{
	if (src.empty()) return false;
	
	const int depth = src.depth();
	const int ch    = src.channels();
	if (depth != CV_8U || (ch != 1 && ch != 3 && ch != 4)) {
		// Only 8-bit 1/3/4 channel images supported here.
		return false;
	}
	
	// Determine target format for ofPixels
	ofPixelFormat fmt;
	switch (ch) {
		case 1: fmt = OF_PIXELS_GRAY; break;
		case 3: fmt = OF_PIXELS_RGB;  break;
		case 4: fmt = OF_PIXELS_RGBA; break;
		default: return false;
	}
	
	// If we need channel order conversion, do it into a temporary Mat that shares the same depth/size.
	cv::Mat converted;
	const cv::Mat* srcPtr = &src;
	
	if (ch == 3 && swapBgrToRgb) {
		cv::cvtColor(src, converted, cv::COLOR_BGR2RGB);
		srcPtr = &converted;
	} else if (ch == 4 && swapBgrToRgb) {
		// OpenCV is BGRA, OF expects RGBA
		cv::cvtColor(src, converted, cv::COLOR_BGRA2RGBA);
		srcPtr = &converted;
	}
	
	// Allocate the ofPixels with desired format/size
	dst.allocate(srcPtr->cols, srcPtr->rows, fmt);
	
	// Fast path if Mat data is contiguous and row stride matches
	const size_t rowBytes = static_cast<size_t>(srcPtr->cols * ch);
	if (srcPtr->isContinuous() &&
		dst.getBytesStride() == rowBytes)
	{
		std::memcpy(dst.getData(), srcPtr->data,
					rowBytes * static_cast<size_t>(srcPtr->rows));
		return true;
	}
	
	// Otherwise, copy row-by-row to respect strides
	const unsigned char* s = srcPtr->data;
	unsigned char* d       = dst.getData();
	const size_t srcStep   = static_cast<size_t>(srcPtr->step);        // bytes per row in Mat
	const size_t dstStep   = static_cast<size_t>(dst.getBytesStride()); // bytes per row in ofPixels
	
	for (int y = 0; y < srcPtr->rows; ++y) {
		std::memcpy(d + y * dstStep, s + y * srcStep, rowBytes);
	}
	return true;
}


//--------------------------------------------------------------
bool ofxRtmPoseDetector::load(std::filesystem::path aOnnxModelPath) {
	return load(aOnnxModelPath, OF_LOG_VERBOSE );
}

//--------------------------------------------------------------
bool ofxRtmPoseDetector::load(std::filesystem::path aOnnxModelPath, ofLogLevel aLogLevel ) {
	ofxOnnx::Settings os;
	#if defined(TARGET_LINUX)
		os.useCuda = true;
	#endif
	os.modelPath = aOnnxModelPath;
	os.envName = "RtmoPoseDetector";
	os.logLevel = aLogLevel;
	
	return loadRtmo(os);
}

//--------------------------------------------------------------
bool ofxRtmPoseDetector::loadRtmo( ofxOnnx::Settings aOnnxSettings ) {
	mBRtmo = true;
	
	if(!mOnnxPose.load(aOnnxSettings)) {
		ofLogError("RtmoPoseDetector::loadRtmo" ) << "Error loading from " << aOnnxSettings.modelPath;
		return false;
	}
	
	if( mOnnxPose.getInputCount() > 0 ) {
		auto shape = mOnnxPose.getInputShape(0);
		if( shape.size() >= 4 ) {
			mDetInputShape = { 1, shape[1], shape[2], shape[3] };
		}
	}
	
	if( mDetInputShape.size() < 1 ) {
		mDetInputShape = { 1, 3, 640, 640 };
	}
	
	mDetInputBuffer.resize(size_t(mDetInputShape[0] * mDetInputShape[1] * mDetInputShape[2] * mDetInputShape[3]));
	
	_addAppListeners();
	
	return true;
}


//--------------------------------------------------------------
bool ofxRtmPoseDetector::load(std::filesystem::path aDetModelPath, std::filesystem::path aPoseModelPath) {
	return load( aDetModelPath, aPoseModelPath, OF_LOG_VERBOSE );
}

//--------------------------------------------------------------
bool ofxRtmPoseDetector::load(std::filesystem::path aDetModelPath, std::filesystem::path aPoseModelPath, ofLogLevel aLogLevel ) {
	ofxOnnx::Settings os;
	#if defined(TARGET_LINUX)
		os.useCuda = true;
	#endif
	os.modelPath = aDetModelPath;
	os.envName = "RtmDetector";
	os.logLevel = aLogLevel;
	if( !loadDetector(os) ) {
		return false;
	}
	
	os.envName = "RtmPose";
	os.modelPath = aPoseModelPath;
	if( !loadPose(os)) {
		return false;
	}
	
	return true;
}

//--------------------------------------------------------------
bool ofxRtmPoseDetector::loadDetector(ofxOnnx::Settings aOnnxSettings ) {
	mBRtmo = false;
	if(!mOnnxDet.load(aOnnxSettings)) {
		ofLogError("RtmoPoseDetector::loadDetector" ) << "Error loading from " << aOnnxSettings.modelPath;
		return false;
	}
	
	if( mOnnxDet.getInputCount() > 0 ) {
		auto shape = mOnnxDet.getInputShape(0);
		if( shape.size() >= 4 ) {
			// in, num channels, height, width
			mDetInputShape = { 1, shape[1], shape[2], shape[3] };
		}
	}
	
	if( mDetInputShape.size() < 1 ) {
		mDetInputShape = { 1, 3, 320, 320 };
	}
	
	mDetInputBuffer.resize(size_t(mDetInputShape[0] * mDetInputShape[1] * mDetInputShape[2] * mDetInputShape[3]));
	return true;
}

//--------------------------------------------------------------
bool ofxRtmPoseDetector::loadPose(ofxOnnx::Settings aOnnxSettings ) {
	mBRtmo = false;
	
	if(!mOnnxPose.load(aOnnxSettings)) {
		ofLogError("RtmoPoseDetector::load" ) << "Error loading from " << aOnnxSettings.modelPath;
		return false;
	}
	_addAppListeners();
	
	return true;
}

//--------------------------------------------------------------
void ofxRtmPoseDetector::update(ofEventArgs& args) {
	// this will be more useful if the data gets threaded at some point
	mBHasNewData = false;
}

//--------------------------------------------------------------
bool ofxRtmPoseDetector::run(ofPixels& apixels) {
	mBHasNewData = false;
	
	mBoxes.clear();
	mPoses.clear();
	
	if( apixels.getWidth() < 3 || apixels.getHeight() < 3 || !apixels.isAllocated() ) {
		ofLogWarning("RtmPoseDetector::run") << "pixels are not allocated.";
		return false;
	}
	
	// doesn't seem to like alpha pixels
//	if( apixels.getNumChannels() != 3 ) {
//		ofLogWarning("RtmPoseDetector::run") << "pixels should be rgb, detected " << apixels.getNumChannels() << " channels.";
//		return false;
//	}
	
	if( mBRtmo ) {
		if( !mOnnxPose.hasSession() ) {
			ofLogWarning("RtmPoseDetector::run") << "mOnnxPose does not have session.";
			return false;
		}
	} else {
		if( !mOnnxDet.hasSession() ) {
			ofLogWarning("RtmPoseDetector::run") << "mOnnxDet does not have session.";
			return false;
		}
	}
	
//	ofLogNotice("ofxRtmPoseDetector") << "run pixels num channels: " << apixels.getNumChannels() << " | " << ofGetFrameNum();
	
	
	cv::Mat cvImg;
	if (apixels.getNumChannels() == 3) {
		cvImg = cv::Mat(apixels.getHeight(), apixels.getWidth(), CV_8UC3, apixels.getData());
		// make a copy, otherwise the cv::Mat will point to the incoming pixels
		mInCvMat = cvImg.clone();
	} else if (apixels.getNumChannels() == 4) {
//		cvImg = cv::Mat(apixels.getHeight(), apixels.getWidth(), CV_8UC4, apixels.getData());
		
		cvImg = cv::Mat(apixels.getHeight(), apixels.getWidth(), CV_8UC4, apixels.getData());
		
		std::vector<cv::Mat> channels;
		cv::split(cvImg, channels);
		
		for (int i = 0; i < 3; i++) {
			// Integer multiply then shift: (src * alpha) >> 8
			// Slightly faster approximation of (src * alpha) / 255
			cv::multiply(channels[i], channels[3], channels[i], 1.0 / 255.0, CV_8U);
		}
		
		cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, mInCvMat);
		
	} else if (apixels.getNumChannels() == 1) {
		cvImg = cv::Mat(apixels.getHeight(), apixels.getWidth(), CV_8UC1, apixels.getData());
		cv::cvtColor(cvImg, mInCvMat, cv::COLOR_GRAY2RGB);
	}
	// make a copy, otherwise the cv::Mat will point to the incoming pixels
//	mInCvMat = cvImg.clone();
	
	if( mSrcRect.width != apixels.getWidth() || mSrcRect.height != apixels.getHeight() ) {
		mDetTex.clear();
		mDetPixels.clear();
	}
	
	mSrcRect.width = apixels.getWidth();
	mSrcRect.height = apixels.getHeight();
	
	if( mOutRect.width < 1 ) {
		mOutRect.width = mSrcRect.width;
	}
	if( mOutRect.height < 1 ) {
		mOutRect.height = mSrcRect.height;
	}
	
	glm::vec2 imgDims = { mSrcRect.width, mSrcRect.height };
	
//	
//	cv::cvtColor(mInCvMat, mInCvMat, cv::COLOR_BGR2RGB);
	
	
	int inW = mDetInputShape[3];
	int inH = mDetInputShape[2];
	
//	float ratio = 1.f;
//	cv::Mat frame_resize;
	auto pp2 = preprocess(mInCvMat, inW, inH);
//	ratio = 1.f / std::get<0>(pp2);
	float ratio = std::get<0>(pp2);
	auto padOffset = std::get<1>(pp2);
//	ofLogNotice("frame_resize") << "dims: " << frame_resize.cols << " x " << frame_resize.rows;
	
	if( mBUseInputTexture ) {
		cvMatToOfPixels( mCvPadded.clone(), mDetPixels, false );
		if( mDetPixels.isAllocated() ) {
			mBDetTexDirty = true;
//			mDetTex.loadData(mDetPixels);
		}
	}
	
	if( ratio < 0.00001f ) {
		ofLogWarning("ofxRtmPoseDetector") << "ratio for preprocess image is too small: " << ratio << " | " << ofGetFrameNum();;
		return false;
	}
	
	
	
	if( mBRtmo ) {
		ratio = std::get<0>(pp2);
		
//		std::vector<Ort::Value> outs = mOnnxPose.run(mDetInputBuffer);
		std::vector<Ort::Value> outs;
		try {
			outs = mOnnxPose.run(mDetInputBuffer);
		} catch (const Ort::Exception& e) {
			// CoreML EP chokes on zero-sized tensors; CPU EP is fine.
			outs.clear();
		}
		
//		ofLogNotice("RtmPoseDetector") << "outs.size(): " << outs.size() << " | " << ofGetFrameNum();
		if( outs.size() > 1 ) {
			auto & dets_val = outs[0];
			auto & kps_val = outs[1];
			
			postprocess_rtmo(dets_val, kps_val, ratio,
							 /*nms_thr=*/0.5f,
							 /*score_thr=*/0.5f);
			
			//ofLogNotice("RtmPoseDetector") << "outs.size(): " << outs.size() << " poses: " << mPoses.size() << " | " << ofGetFrameNum();
			
			if( getRemoveOverlappingPoses() ) {
				mPoses = _removeOverlappingPoses(mPoses, mPoseOverlapThreshToRemove);
			}
		} else {
			
		}
	} else {
		ratio = 1.f / std::get<0>(pp2);
		
//		std::vector<Ort::Value> outs = mOnnxDet.run(mDetInputBuffer);
//		getInputCpuTensor(int aInputIndex, std::vector<T>& adata, std::vector<int64_t> aShape );
		auto inputShape = mOnnxDet.getInputShape(0);
//		for( size_t is = 0; is < inputShape.size(); is++ ) {
//			ofLogNotice("RtmPoseDetector") << is << " value: " << inputShape[is] << " | " << ofGetFrameNum();
//		}
		auto inTensor = mOnnxDet.getInputCpuTensor(0, mDetInputBuffer, inputShape );
		
//		auto& t = input_tensor; // Ort::Value
		auto info = inTensor.GetTensorTypeAndShapeInfo();
		auto shape = info.GetShape();
		int64_t elems = 1;
		for (auto d : shape) {
			if (d <= 0) {
//				throw std::runtime_error("empty/bad input dim");
				ofLogError("RtmPoseDetector") << "run: empty/bad input dim.";
			} else {
				elems *= d;
			}
		}
		
		std::vector<Ort::Value> outs;
		try {
			outs = mOnnxDet.run(std::move(inTensor));
		} catch (const Ort::Exception& e) {
			// CoreML EP chokes on zero-sized tensors; CPU EP is fine.
			outs.clear();
		}
		
		ofLogVerbose("RtmPoseDetector") << "outs: " << outs.size();
		
		if( outs.size() > 1 ) {
			
			auto& boxes_val = outs[0];
			auto& labels_val = outs[1];
			
			auto boxesInfo = boxes_val.GetTensorTypeAndShapeInfo();
			auto boxesShape = boxesInfo.GetShape(); // [1, dynamic, 5]
			
			auto labelsInfo = labels_val.GetTensorTypeAndShapeInfo();
			auto labelsShape = labelsInfo.GetShape(); // [1, dynamic]
			
			const float* boxesData = boxes_val.GetTensorData<float>();
			const int* labelsData = labels_val.GetTensorData<int>();
			
			// lets check that we are getting what we are expecting.
			if( boxesShape.size() != 3 || labelsShape.size() != 2 ) {
				ofLogError("RtmPoseDetector::run") << "Unexpected shapes: dets: " << boxesShape.size() << " labels: " << labelsShape.size();
				return false;
			}
			
			// check that the number of boxes is equal to the nubmer of labels
			if( boxesShape[1] != labelsShape[1] ) {
				ofLogError("RtmPoseDetector::run") << "Dets and labels size mismatch: " << boxesShape[1] << " labels: " << labelsShape[1];
				return false;
			}

			// ofLogNotice("RtmPoseDetector::run") << "boxes: " << boxesShape[1] << " | " << ofGetFrameNum();

			// offset for the size of the data
			auto boxIndexOffset = boxesShape[2];
			
			// now lets get some boxes!
			int total = boxesShape[1];
			for( int j = 0; j < total; j++ ) {
				// do we have a label?
				int label = labelsData[j];
				if( label != 0 ) {
					ofLogError("RtmPoseDetector") << "label is " << label << " | " << ofGetFrameNum();
					continue;
				}
				
				int joff = boxIndexOffset * j;
				ofxRtmPose::Box b;
				float x1 = boxesData[joff + 0];// * ratio;
				float y1 = boxesData[joff + 1];// * ratio;
				float x2 = boxesData[joff + 2];// * ratio;
				float y2 = boxesData[joff + 3];// * ratio;
				b.score = boxesData[joff + 4 ];
				b.labelIndex = labelsData[j];
				
				b.srcRect.x = x1;
				b.srcRect.y = y1;
				b.srcRect.width = x2-x1;
				b.srcRect.height = y2-y1;
				
				b.outRect = b.srcRect;
				b.outRect.x -= padOffset.x;
				b.outRect.y -= padOffset.y;
				b.outRect.scale(ratio);
				b.outRect.x *= ratio;
				b.outRect.y *= ratio;
				
				//			ofLogNotice("boxes") << j << " - score: " << b.score;
				
				if( b.score < mBoxThresh || b.srcRect.width < 1 || b.srcRect.height < 1 ) {
					// if( b.srcRect.width < 1 || b.srcRect.height < 1 ) {
					ofLogVerbose("RtmPoseDetector") << "rejecting box " << b.score << " src rect: " << b.srcRect << " | " << ofGetFrameNum();
					continue;
				}
				
				mBoxes.push_back(b);
			}

			// ofLogNotice("RtmPoseDetector::run") << "filtered boxes: " << mBoxes.size() << " | " << ofGetFrameNum();

			// maybe we should sort the boxes?
			//TODO: SORT BOXES BASED ON SCORE
			//TODO: MERGE BOXES with overlap score
		}
		
				
		if(mBAddBoxIfNoneDetected && mBoxes.size() < 1 ) {
			ofxRtmPose::Box box;
			box.outRect.width = mInCvMat.cols;
			box.outRect.height = mInCvMat.rows;
			box.score = 0.f;
			mBoxes.push_back(box);
		}
		
		mPoses.clear();
		postProcess(mInCvMat, mBoxes, mPoses );
		if( getRemoveOverlappingPoses() ) {
			mPoses = _removeOverlappingPoses(mPoses, mPoseOverlapThreshToRemove);
		}
	}
	
	mBHasNewData = true;
	
//	return mPoses.size() > 0;
	return true;
}



//--------------------------------------
void ofxRtmPoseDetector::draw() {
	
	if(mBUseInputTexture) {
		if( mBDetTexDirty ) {
			mBDetTexDirty = false;
			mDetTex.loadData(mDetPixels);
		}
		
		ofPushMatrix(); {
			ofTranslate(mSrcRect.width+10, 0.f);
			if( mDetTex.isAllocated() ) {
				ofSetColor(255);
				mDetTex.draw(0,0);
			}
			
			for( int i = 0; i < mBoxes.size(); i++ ) {
				
				auto brect = mBoxes[i].srcRect;
				ofNoFill();
				ofSetColor(ofColor::green);
				ofDrawRectangle( brect );
				ofFill();
				
				ofDrawBitmapString(ofToString(mBoxes[i].score) + " i: "+ofToString(mBoxes[i].labelIndex), brect.x+16, brect.y+16 );
			}
		} ofPopMatrix();
	}
	
	
	
//	ofNoFill();
	int ri = 0;
	std::vector<ofColor> drawColors = {ofColor::yellow, ofColor::green, ofColor::magenta, ofColor::cyan };
	for( auto& result : mPoses ) {
		result.drawWithLinkColors(mDrawThresh);
		ofSetColor( drawColors[ri%drawColors.size()]);
		for( std::size_t i = 0; i < result.keypoints.size(); i++ ) {
			if( result.scores[i] < mDrawThresh ) continue;
			ofDrawCircle( result.keypoints[i], 2.f );
//			ofDrawBitmapStringHighlight(ofToString(result.scores[i],2), result.keypoints[i] );
		}
		
		ofNoFill();
		ofDrawRectangle(result.box.outRect);
//		ofSetColor( ofColor::blue );
//		ofDrawRectangle(result.scaledRect);
		ofFill();
		ri++;
	}
//	ofFill();
	
//	for( int i = 0; i < mBoxes.size(); i++ ) {
//		auto brect = mBoxes[i].outRect;
//		
//		ofNoFill();
//		ofSetColor(ofColor::green);
//		ofDrawRectangle( brect );
//		ofFill();
//		
//		ofDrawBitmapString(ofToString(mBoxes[i].score) + " i: "+ofToString(mBoxes[i].labelIndex), brect.x+16, brect.y+16 );
//	}
}

// HWC RGB uint8 -> NCHW float32 (0..255) without cv::split
void ofxRtmPoseDetector::hwc_to_nchw_rgb_u8_to_f32(const cv::Mat & img, float * outNCHW) {
	const int H = img.rows, W = img.cols;
	const size_t plane = size_t(H) * W;
	float * r = outNCHW + 0 * plane;
	float * g = outNCHW + 1 * plane;
	float * b = outNCHW + 2 * plane;
	
	
	const float s = 1.0f / 255.0f;
	
	if( mBRtmo ) {
		for (int y = 0; y < H; ++y) {
			const uint8_t * p = img.ptr<uint8_t>(y);
			for (int x = 0; x < W; ++x) {
				*r++ = float(p[3 * x + 0]);
				*g++ = float(p[3 * x + 1]);
				*b++ = float(p[3 * x + 2]);
			}
		}
	} else {
		
		for (int y = 0; y < H; ++y) {
			const uint8_t * p = img.ptr<uint8_t>(y);
			
//			for (int x = 0; x < W; ++x) {
//				*r++ = (float(p[3 * x + 0]) * s);
//				*g++ = (float(p[3 * x + 1]) * s);
//				*b++ = (float(p[3 * x + 2]) * s);
//			}
			
			for (int x = 0; x < W; ++x) {
				*r++ = ((float(p[3 * x + 0]) - IMAGE_MEAN[0]) / IMAGE_STD[0]);
				*g++ = ((float(p[3 * x + 1]) - IMAGE_MEAN[1]) / IMAGE_STD[1]);
				*b++ = ((float(p[3 * x + 2]) - IMAGE_MEAN[2]) / IMAGE_STD[2]);
			}
		}
	}
}

//--------------------------------------------------------------------
std::tuple<float, glm::vec2> ofxRtmPoseDetector::preprocess(const cv::Mat & frame_rgb, int aw, int ah) {
	const int src_h = frame_rgb.rows, src_w = frame_rgb.cols;

	int dst_w = aw;
	int dst_h = ah;

	const float r = std::min(float(dst_h) / float(src_h), float(dst_w) / float(src_w));

	const int new_w = int(src_w * r);
	const int new_h = int(src_h * r);

	bool normalize_0_1 = false;
	
	if( new_w < 1 || new_h < 1 ) {
		return { 0.f, glm::vec2(0,0) };
	}

	// Resize
//	cv::resize(frame_rgb, mCvResized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
//
//	if( mCvPadded.empty() || mCvPadded.cols != aw || mCvPadded.rows != ah ) {
//		mCvPadded.create(dst_h, dst_w, CV_8UC3);
//	}
//
//	mCvPadded.setTo(cv::Scalar(114, 114, 114));
//	// Pad (top-left) with 114
////	cv::Mat padded(dst_h, dst_w, CV_8UC3, cv::Scalar(114,114,114));
//	mCvResized.copyTo(mCvPadded(cv::Rect(0, 0, new_w, new_h)));
	
	// 1) Ensure mCvResized is a normal mat (not an ROI view) before using as resize dst
	if (mCvResized.isSubmatrix()) {
		mCvResized.release(); // or mCvResized = mCvResized.clone();
	}
	
	// 2) Resize into mCvResized (this is fine because it's a full Mat, not an ROI)
	cv::resize(frame_rgb, mCvResized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
	
	if( mCvResized.empty() ) {
		return { 0.f, glm::vec2(0,0) };
	}
	
	// 3) Create padded to match size + TYPE of resized (channels matter!)
	if (mCvPadded.empty() ||
		mCvPadded.cols != dst_w ||
		mCvPadded.rows != dst_h ||
		mCvPadded.type() != mCvResized.type()) {
		
		mCvPadded.create(dst_h, dst_w, mCvResized.type());
	}
	
	// 4) Fill padding with 114 (respect channel count)
	if (mCvPadded.channels() == 4) {
		mCvPadded.setTo(cv::Scalar(114, 114, 114, 255));
	} else if (mCvPadded.channels() == 3) {
		mCvPadded.setTo(cv::Scalar(114, 114, 114));
	} else if (mCvPadded.channels() == 1) {
		mCvPadded.setTo(cv::Scalar(114));
	} else {
		// unusual channel counts — fall back safely
		mCvPadded.setTo(cv::Scalar::all(114));
	}
	
	// 5) Compute ROI and clamp it to the padded bounds (prevents ROI assert)
	cv::Rect roi(0, 0, new_w, new_h);
	roi &= cv::Rect(0, 0, mCvPadded.cols, mCvPadded.rows);
	
	if (roi.width > 0 && roi.height > 0) {
		// Also clamp source in case roi got reduced
		cv::Rect srcRoi(0, 0, roi.width, roi.height);
		try {
			mCvResized(srcRoi).copyTo(mCvPadded(roi)); // safe: same type, in-bounds ROI
		} catch(const cv::Exception& e) {
			
			ofLogError("OpenCV ROI") << "---- ROI CRASH INFO ----";
			ofLogError("OpenCV ROI") << "new_w=" << new_w << " new_h=" << new_h;
			
			ofLogError("OpenCV ROI") << "mCvResized: " << mCvResized.cols << "x" << mCvResized.rows << " channels=" << mCvResized.channels();
			
			ofLogError("OpenCV ROI") << "mCvPadded: " << mCvPadded.cols << "x" << mCvPadded.rows << " channels=" << mCvPadded.channels();
			
			ofLogError("OpenCV ROI") << e.what();
			
			return { 0.f, glm::vec2(0,0) };
		}
	}
	

	// HWC (RGB) -> CHW float32
	const size_t plane = size_t(dst_h) * dst_w;
	float* rplane = mDetInputBuffer.data() + 0 * plane;
	float* gplane = mDetInputBuffer.data() + 1 * plane;
	float* bplane = mDetInputBuffer.data() + 2 * plane;

	const float s = normalize_0_1 ? (1.0f / 255.0f) : 1.0f;

	for (int y = 0; y < dst_h; ++y) {
		const uint8_t* row = mCvPadded.ptr<uint8_t>(y);
		for (int x = 0; x < dst_w; ++x) {
			// RGB order
			const uint8_t R = row[3*x + 0];
			const uint8_t G = row[3*x + 1];
			const uint8_t B = row[3*x + 2];
			*rplane++ = R * s;
			*gplane++ = G * s;
			*bplane++ = B * s;
		}
	}

	//	hwc_to_nchw_bgr_u8_to_f32(mCvPadded, mDetInputBuffer.data());

	return { r, glm::vec2(0,0) };
}

// IoU for [x1,y1,x2,y2]
inline float ofxRtmPoseDetector::iou(const TBox& a, const TBox& b) {
	float xx1 = std::max(a.x1, b.x1);
	float yy1 = std::max(a.y1, b.y1);
	float xx2 = std::min(a.x2, b.x2);
	float yy2 = std::min(a.y2, b.y2);
	float w = std::max(0.0f, xx2 - xx1);
	float h = std::max(0.0f, yy2 - yy1);
	float inter = w * h;
	float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
	float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
	float uni = areaA + areaB - inter + 1e-6f;
	return inter / uni;
}

// Single-class greedy NMS (indices into the input arrays)
inline std::vector<int> ofxRtmPoseDetector::nms(const std::vector<TBox> & boxes, float nms_thr, float score_thr) {
	std::vector<int> idx(boxes.size());
	std::iota(idx.begin(), idx.end(), 0);
	// filter by score
	idx.erase(std::remove_if(idx.begin(), idx.end(),
							 [&](int i) { return boxes[i].score < score_thr; }),
			  idx.end());
	// sort by score desc
	std::sort(idx.begin(), idx.end(), [&](int a, int b) { return boxes[a].score > boxes[b].score; });
	
	std::vector<int> keep;
	for (int i = 0; i < (int)idx.size(); ++i) {
		bool suppressed = false;
		for (int j : keep) {
			if (iou(boxes[idx[i]], boxes[j]) > nms_thr) {
				suppressed = true;
				break;
			}
		}
		if (!suppressed) keep.push_back(idx[i]);
	}
	return keep;
}

//-------------------------------------------------------------------------------
void ofxRtmPoseDetector::postprocess_rtmo(const Ort::Value & dets_val,
										const Ort::Value & kps_val,
										float ratio,
										std::optional<float> nms_thr,
										std::optional<float> score_thr) {
	
	mPoses.clear();
	
	// ----- shapes -----
	auto dets_info = dets_val.GetTensorTypeAndShapeInfo();
	auto kps_info = kps_val.GetTensorTypeAndShapeInfo();
	auto dets_shape = dets_info.GetShape(); // [N, M, D]
	auto kps_shape = kps_info.GetShape(); // [N, M, K, C]
	
	int64_t N = dets_shape.size() > 0 ? dets_shape[0] : 1;
	int64_t M = dets_shape.size() > 1 ? dets_shape[1] : 0;
	int64_t D = dets_shape.size() > 2 ? dets_shape[2] : 0;
	int64_t NK = kps_shape.size() > 0 ? kps_shape[0] : 1;
	int64_t MK = kps_shape.size() > 1 ? kps_shape[1] : 0;
	int64_t K = kps_shape.size() > 2 ? kps_shape[2] : 0;
	int64_t C = kps_shape.size() > 3 ? kps_shape[3] : 0;
	
	// We assume N == NK == 1
	(void)N;
	(void)NK;
	
	const float * det = dets_val.GetTensorData<float>();
	const float * kps = kps_val.GetTensorData<float>();
	
	// ----- slice dets: boxes [0,:,0:4], scores [0,:,4] -----
	std::vector<TBox> boxes;
	boxes.resize((size_t)M);
	for (int64_t m = 0; m < M; ++m) {
		size_t base = (size_t)m * (size_t)D; // N=1
		TBox b;
		if (D >= 4) {
			b.x1 = det[base + 0] / ratio;
			b.y1 = det[base + 1] / ratio;
			b.x2 = det[base + 2] / ratio;
			b.y2 = det[base + 3] / ratio;
		}
		b.score = (D >= 5) ? det[base + 4] : 1.0f;
		boxes[(size_t)m] = b;
	}
	
	// ----- slice keypoints: [0,:,:,:2] and scores [0,:,:,2] -----
	std::vector<std::vector<glm::vec2>> keypoints((size_t)M, std::vector<glm::vec2>((size_t)K));
	std::vector<std::vector<float>> kp_scores((size_t)M, std::vector<float>((size_t)K, 1.0f));
	
	for (int64_t m = 0; m < M; ++m) {
		for (int64_t k = 0; k < K; ++k) {
			size_t idx = (size_t)((m * K + k) * C); // N=1
			float x = (C >= 1) ? kps[idx + 0] / ratio : 0.0f;
			float y = (C >= 2) ? kps[idx + 1] / ratio : 0.0f;
			keypoints[(size_t)m][(size_t)k] = glm::vec2(x, y);
			if (C >= 3) kp_scores[(size_t)m][(size_t)k] = kps[idx + 2];
		}
	}
	
	// ----- NMS (and score threshold) -----
	const float nmsT = nms_thr.value_or(0.65f); // pick sensible defaults if not given
	const float scoreT = score_thr.value_or(0.25f);
	
	bool b_nms = false;
	
	std::vector<int> keep;
	if( b_nms ) {
		keep = nms(boxes, nmsT, scoreT);
	} else {
		keep.reserve((size_t)M);
		for (int64_t m = 0; m < M; ++m)
			if (boxes[(size_t)m].score >= scoreT) keep.push_back((int)m);
	}
	
	// If nothing kept, return a single all-zero pose
	if (keep.empty()) {
		std::vector<std::vector<glm::vec2>> kp1(1, std::vector<glm::vec2>((size_t)K, glm::vec2(0, 0)));
		std::vector<std::vector<float>> sc1(1, std::vector<float>((size_t)K, 0.0f));
		std::vector<TBox> bx1(1, TBox { 0, 0, 0, 0, 0 });
		// std::vector<int> idx; // empty
		// return { std::move(kp1), std::move(sc1), std::move(bx1), std::move(idx) };
		//		return { std::move(kp1), std::move(sc1), std::move(bx1) };
		return;
	}
	
	// Filter by keep indices
	auto gather_nested_vec2 = [&](const std::vector<std::vector<glm::vec2>> & in) {
		std::vector<std::vector<glm::vec2>> out;
		out.reserve(keep.size());
		for (int i : keep)
			out.push_back(in[(size_t)i]);
		return out;
	};
	auto gather_nested_flt = [&](const std::vector<std::vector<float>> & in) {
		std::vector<std::vector<float>> out;
		out.reserve(keep.size());
		for (int i : keep)
			out.push_back(in[(size_t)i]);
		return out;
	};
	auto gather_boxes = [&](const std::vector<TBox> & in) {
		std::vector<TBox> out;
		out.reserve(keep.size());
		for (int i : keep)
			out.push_back(in[(size_t)i]);
		return out;
	};
	
	auto f_keypoints = gather_nested_vec2(keypoints);
	auto f_scores = gather_nested_flt(kp_scores);
	auto f_boxes = gather_boxes(boxes);
	
	//ofLogNotice("ofxRtmPoseDetector::postprocess_rtmo") << "keypoints.size(): " << f_keypoints.size() << " scores: " << f_scores.size() << " boxes: " << f_boxes.size() << " | " << ofGetFrameNum();
	std::size_t num = std::min(f_keypoints.size(), std::min(f_scores.size(), f_boxes.size()));
	for (size_t i = 0; i < num; ++i) { // i = kept pose index
		ofxRtmPose tpose;
		tpose.keypoints = f_keypoints[i];
		tpose.box.score = f_boxes[i].score;
		
		tpose.box.srcRect = ofRectangle(f_boxes[i].x1, f_boxes[i].y1, f_boxes[i].x2 - f_boxes[i].x1, f_boxes[i].y2 - f_boxes[i].y1);
		tpose.box.outRect = tpose.box.srcRect;
		tpose.scores = f_scores[i];
		
		mapPoseToOutRect(tpose);
		
		mPoses.push_back(tpose);
		mPoses.back().configSpec();
	}
}

//--------------------------------------------------------------------
void ofxRtmPoseDetector::postProcess(const cv::Mat& frame_rgb, std::vector<ofxRtmPose::Box>& aboxes, std::vector<ofxRtmPose>& aresults ) {
	
	std::array<int64_t, 4> inputShape = {1,3,256,256};
	if( mOnnxPose.getInputCount() > 0 ) {
		auto shape = mOnnxPose.getInputShape(0);
		if( shape.size() >= 4 ) {
			inputShape = { 1, shape[1], shape[2], shape[3] };
		}
	}
	
	int inW = inputShape[3];
	int inH = inputShape[2];
	
	int srcw = frame_rgb.cols;
	int srch = frame_rgb.rows;
	
	std::vector<float> qbuffer(size_t(inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3]));
	
//	cv::Mat cpadded;
//	cpadded.create(inH, inW, CV_8UC3);
	cv::Mat cpadded;
	cpadded.create(inH, inW, CV_8UC3);
	
	int boxIndex = 0;
	float boxScale = 1.25f;
	for( auto& box : aboxes ) {
		auto scaledBoxRect = box.outRect;
		scaledBoxRect.scaleFromCenter(boxScale);
		float x1 = scaledBoxRect.x;
		float x2 = scaledBoxRect.getRight();
		float y1 = scaledBoxRect.y;
		float y2 = scaledBoxRect.getBottom();
		
		x1 = ofClamp(x1, 0, srcw-1);
		x2 = ofClamp(x2, 0, srcw-1);
		y1 = ofClamp(y1, 0, srch-1);
		y2 = ofClamp(y2, 0, srch-1);
		
		scaledBoxRect.x = x1;
		scaledBoxRect.y = y1;
		scaledBoxRect.width = x2-x1;
		scaledBoxRect.height = y2-y1;
		
		if( scaledBoxRect.width < 1 || scaledBoxRect.height < 1 ) {
			ofLogError("RtmPoseDetector::postProcess") << boxIndex << " - " << scaledBoxRect.width << " x " << scaledBoxRect.height;
			continue;
		}
		
		if( scaledBoxRect.x > srcw-2 ) {
			ofLogError("RtmPoseDetector::postProcess") << boxIndex << " - " << scaledBoxRect.x << " > " << (srcw-2);
			continue;
		}
		if( scaledBoxRect.y > srch-2 ) {
			ofLogError("RtmPoseDetector::postProcess") << boxIndex << " - " << scaledBoxRect.y << " > " << (srch-2);
			continue;
		}
		
		
		cv::Mat inputRoi;// = frame_rgb(cv::Rect(scaledBoxRect.x, scaledBoxRect.y, scaledBoxRect.width, scaledBoxRect.height));
		try {
			inputRoi = frame_rgb(cv::Rect(scaledBoxRect.x, scaledBoxRect.y, scaledBoxRect.width, scaledBoxRect.height));
		} catch(const cv::Exception& e) {
			ofLogError() << "CV Error: frame_rgb: " << frame_rgb.cols << " x " <<frame_rgb.rows;
			ofLogError("ofxRtmPoseDetector::postProcess") << "CV input roi error: " << scaledBoxRect;
			ofLogError("ofxRtmPoseDetector::postProcess") << "CV Error: " << e.what();
			continue;
		}
		cv::Mat inputCopy = inputRoi.clone();
		// ok we have a copy of the image in the original image space
		// now we need to scale it down into the pose input dimensions //
		
		cpadded.setTo(cv::Scalar(128, 128, 128));
		
		float r = std::min(float(inH) / float(inputCopy.rows), float(inW) / float(inputCopy.cols));
		int nh = std::lround(inputCopy.rows * r);
		int nw = std::lround(inputCopy.cols * r);
		
		int dx = (inW - nw) / 2;
		int dy = (inH - nh) / 2;
		
		cv::Mat cresized;
		cv::resize(inputCopy, cresized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
		cresized.copyTo(cpadded(cv::Rect(dx, dy, nw, nh)));
		
		ofLogVerbose("RtmPoseDetector::postProcess" ) << "in: " << inW << " x " << inH << " nw: " << nw << " x " << nh;
		
		hwc_to_nchw_rgb_u8_to_f32(cpadded, qbuffer.data());
		
		
		
//		Ort::Value input = Ort::Value::CreateTensor<float>(mOnnxPose.getCpuMem(),
//														   qbuffer.data(), qbuffer.size(),
//														   inputShape.data(), inputShape.size());
//		
//		std::vector<const char *> in_names = { "input" };
//		std::vector<const char *> out_names = { "simcc_x", "simcc_y" };
//		std::array<Ort::Value, 1> inputs = { std::move(input) };
//		
//		std::vector<Ort::Value> outs;
//		{
//			outs = mOnnxPose.getSession().Run(Ort::RunOptions { nullptr },
//											 in_names.data(), inputs.data(), in_names.size(),
//											 out_names.data(), out_names.size());
//		}
		
//		std::vector<Ort::Value> outs = mOnnxPose.run(qbuffer);
		std::vector<Ort::Value> outs;
		try {
			outs = mOnnxPose.run(qbuffer);
		} catch (const Ort::Exception& e) {
			// CoreML EP chokes on zero-sized tensors; CPU EP is fine.
			outs.clear();
		}
		
		
		if( outs.size() > 1 ) {
			auto& simcc_x_val = outs[0];
			auto& simcc_y_val = outs[1];
			
			auto simcc_x_Info = simcc_x_val.GetTensorTypeAndShapeInfo();
			auto simcc_x_Shape = simcc_x_Info.GetShape(); // [dynamic, dynamic, 512]
			
			auto simcc_y_Info = simcc_y_val.GetTensorTypeAndShapeInfo();
			auto simcc_y_Shape = simcc_y_Info.GetShape(); // [dynamic, dynamic, 512]
			
			// simcc calculation from https://github.com/HW140701/RTMPose-Deploy/tree/main/Windows/OnnxRumtime-CPU/src/RTMPoseOnnxRuntime
			float* simcc_x_result = outs[0].GetTensorMutableData<float>();
			float* simcc_y_result = outs[1].GetTensorMutableData<float>();
			
			if(simcc_x_Shape.size() != 3 || simcc_y_Shape.size() != 3 ) {
				ofLogError("RtmPoseDetector::postProcess") << boxIndex << " - x: " << simcc_x_Shape.size() << " y: " << simcc_y_Shape.size();
				continue;
			}
			
			// now we can move on
			int extend_width = simcc_x_Shape[2];
			int extend_height = simcc_y_Shape[2];
			
			int batch_size = simcc_x_Shape[0] == simcc_y_Shape[0] ? simcc_x_Shape[0] : 0;
			int joint_num = simcc_x_Shape[1] == simcc_y_Shape[1] ? simcc_x_Shape[1] : 0;
			
			
//			if( batch_size > 1 || joint_num > 17 ) {
//				ofLogNotice("RtmPoseDetector") << "batch_size: " << batch_size << " joint_num: " << joint_num;
//			}
			
			ofxRtmPose poseR;
			for (int i = 0; i < joint_num; ++i) {
				
				// find the maximum and maximum indexes in the value of each Extend_width length
				auto x_biggest_iter = std::max_element(simcc_x_result + i * extend_width, simcc_x_result + i * extend_width + extend_width);
				int max_x_pos = std::distance(simcc_x_result + i * extend_width, x_biggest_iter);
				int pose_x = max_x_pos / 2;
				float score_x = *x_biggest_iter;
				
				// find the maximum and maximum indexes in the value of each exten_height length
				auto y_biggest_iter = std::max_element(simcc_y_result + i * extend_height, simcc_y_result + i * extend_height + extend_height);
				int max_y_pos = std::distance(simcc_y_result + i * extend_height, y_biggest_iter);
				int pose_y = max_y_pos / 2;
				float score_y = *y_biggest_iter;
				
				//float score = (score_x + score_y) / 2;
				float score = std::max(score_x, score_y);
				
				
				auto kp = glm::vec2( float(pose_x), float(pose_y) );
				kp.x -= dx;
				kp.y -= dy;
				kp.x /= r;
				kp.y /= r;
				kp.x += scaledBoxRect.x;// / r;
				kp.y += scaledBoxRect.y;// / r;
				poseR.keypoints.push_back(kp);
				poseR.scores.push_back(score);
				poseR.score += score;
//				poseR.scaledRect = scaledBoxRect;
				poseR.box = box;
				
			}
			
			if( poseR.keypoints.size() > 0 ) {
				
				if( poseR.scores.size() > 1 ) {
					poseR.score /= float(poseR.scores.size());
				}
				
				// filter by min score
				if( poseR.score > 0.3f ) {
					mapPoseToOutRect(poseR);
					
					ofLogVerbose("PoseR num keypoints: ") << poseR.keypoints.size() << " | " << ofGetFrameNum();
					mPoses.push_back(poseR);
					mPoses.back().configSpec();
					mPoses.back()._updateHands();
				}
			}
		}
		
		
		
		boxIndex++;
	}
}

//--------------------------------------------------------------------
void ofxRtmPoseDetector::mapPoseToOutRect( ofxRtmPose& apose ) {
	// remap keypoints to out rect
	for( auto& kp : apose.keypoints ) {
		kp.x = (kp.x / mSrcRect.width) * mOutRect.width + mOutRect.x;
		kp.y = (kp.y / mSrcRect.height) * mOutRect.height + mOutRect.y;
	}
	if( mSrcRect.width > 0 && mSrcRect.height > 0 ) {
		if( apose.box.outRect.width > 0 ) {
			apose.box.outRect.width *= ( mOutRect.width / mSrcRect.width );
		}
		if( apose.box.outRect.height > 0 ) {
			apose.box.outRect.height *= ( mOutRect.height / mSrcRect.height );
		}
		apose.box.outRect.x = (apose.box.outRect.x / mSrcRect.width) * mOutRect.width + mOutRect.x;
		apose.box.outRect.y = (apose.box.outRect.y / mSrcRect.height) * mOutRect.height + mOutRect.y;
	}
}

//--------------------------------------------------------------------
std::vector<ofxRtmPose> ofxRtmPoseDetector::_removeOverlappingPoses( const std::vector<ofxRtmPose>& aposes, float aiouThreshold ) {
	if( aposes.size() < 2 ) {
		return aposes;
	}
	
	// Sort by confidence descending — keep the more confident detection
	std::vector<ofxRtmPose> sorted = aposes;
	std::sort(sorted.begin(), sorted.end(), [](const ofxRtmPose& a, const ofxRtmPose& b) {
		return a.getScore() > b.getScore();
	});
	
	std::vector<bool> suppressed(sorted.size(), false);
	
	for (int i = 0; i < sorted.size(); i++) {
		if (suppressed[i]) {continue;}
		ofRectangle bboxI = sorted[i].getBoundingBox();
		
		for (int j = i + 1; j < sorted.size(); j++) {
			if (suppressed[j]) {continue;}
			ofRectangle bboxJ = sorted[j].getBoundingBox();
			
			if (ofxRtmPoseDetector::sComputeIoU(bboxI, bboxJ) > aiouThreshold) {
				suppressed[j] = true;
			}
		}
	}
	
	std::vector<ofxRtmPose> result;
	for (int i = 0; i < sorted.size(); i++) {
		if (!suppressed[i]) {
			result.push_back(sorted[i]);
		}
	}
	
	return result;
	
}

//--------------------------------------------------------------
void ofxRtmPoseDetector::_onExit( ofEventArgs& args ) {
	_removeAppListeners();
}

//--------------------------------------------------------------
void ofxRtmPoseDetector::_addAppListeners() {
	if( !mBHasAppListeners ) {
		ofAddListener(ofEvents().update, this, &ofxRtmPoseDetector::update );
		ofAddListener(ofEvents().exit, this, &ofxRtmPoseDetector::_onExit, OF_EVENT_ORDER_BEFORE_APP );
	}
	mBHasAppListeners=true;
}

//------------------------------------------------------------------------
void ofxRtmPoseDetector::_removeAppListeners() {
	if(mBHasAppListeners) {
		ofRemoveListener(ofEvents().update, this, &ofxRtmPoseDetector::update );
		ofRemoveListener(ofEvents().exit, this, &ofxRtmPoseDetector::_onExit, OF_EVENT_ORDER_BEFORE_APP );
	}
	mBHasAppListeners=false;
}

//------------------------------------------------------------------------
float ofxRtmPoseDetector::sComputeIoU(const ofRectangle& a, const ofRectangle& b) {
	ofRectangle inter = a.getIntersection(b); // oF does this for you
	float interArea = inter.getArea();
	if (interArea == 0.0f) return 0.0f;
	return interArea / (a.getArea() + b.getArea() - interArea);
}
