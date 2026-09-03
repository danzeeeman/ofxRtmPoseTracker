
#pragma once
#include "ofMesh.h"
#include "ofxRtmSkeletonSpec.h"

class ofxRtmHand {
public:
	
	enum Handedness : int {
		RIGHT=0,
		LEFT
	};
	
	void updateFromHand( const ofxRtmHand& ahand );
	void updateFromHandWithSmoothing( const ofxRtmHand& ahand, float aSmoothing );
	bool isKeypointValid( int aindex, float athresh );
	bool isKeypointValid( int aindex, float athresh ) const;
	
	ofRectangle getBoundingBox() const;

	void draw(float athresh);
	
	std::vector<glm::vec2> keypoints;
	std::vector<float> scores;
	float score = 0.f;

	float getScore();
	float getScore() const;
	
	float fingerLength(const std::vector<glm::vec2>& kps, int start, int end);
	bool hasPlausibleFingerLengths(float aMinHandSize);
	
	Handedness handed = Handedness::RIGHT;
	
	glm::vec2 getPalmPos();
	glm::vec2 getPalmPos() const;
	
protected:
	void _checkSkel();
	
	ofxRtmSkeletonSpec* mSkelSpecPtr = nullptr;
};
