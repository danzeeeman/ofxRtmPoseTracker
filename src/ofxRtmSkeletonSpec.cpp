
#include "ofxRtmSkeletonSpec.h"

std::unordered_map<std::string, ofxRtmSkeletonSpec> ofxRtmSkeletonSpecRegistry::sSkelRegistry;

//--------------------------------------------------------------
void ofxRtmSkeletonSpecRegistry::sRegisterSkeletonSpec(const std::string& aname, const ofxRtmSkeletonSpec& s) {
	sSkelRegistry[aname] = s;
}

//--------------------------------------------------------------
bool ofxRtmSkeletonSpecRegistry::sHasSkeletonSpec(const std::string& aname) {
	return sSkelRegistry.contains(aname);
}

//--------------------------------------------------------------
ofxRtmSkeletonSpec* ofxRtmSkeletonSpecRegistry::sGetSkeletonSpec(const std::string& aname) {
	auto it = sSkelRegistry.find(aname);
	return (it == sSkelRegistry.end()) ? nullptr : &it->second;
}

// -----------------------------------------
// WholeBody133 (body+feet+face+both hands)
// -----------------------------------------
ofxRtmSkeletonSpec ofxRtmSkeletonSpec::getWholebody133Spec() {
	ofxRtmSkeletonSpec S;
	S.name = "wholebody133";
	auto& N = S.names;
	N.resize(133);
	
	// 0–16 body
	N[0]="nose"; N[1]="left_eye"; N[2]="right_eye"; N[3]="left_ear"; N[4]="right_ear";
	N[5]="left_shoulder"; N[6]="right_shoulder"; N[7]="left_elbow"; N[8]="right_elbow";
	N[9]="left_wrist"; N[10]="right_wrist"; N[11]="left_hip"; N[12]="right_hip";
	N[13]="left_knee"; N[14]="right_knee"; N[15]="left_ankle"; N[16]="right_ankle";
	// 17–22 feet
	N[17]="left_big_toe"; N[18]="left_small_toe"; N[19]="left_heel";
	N[20]="right_big_toe"; N[21]="right_small_toe"; N[22]="right_heel";
	// 23–90 face (68)
	for (int i=23;i<=90;++i) N[i] = "face_"+std::to_string(i-23);
	// 91–111 left hand (21), 112–132 right hand (21)
	N[91]="left_wrist"; N[92]="left_thumb1"; N[93]="left_thumb2"; N[94]="left_thumb3"; N[95]="left_thumb4";
	N[96]="left_index1"; N[97]="left_index2"; N[98]="left_index3"; N[99]="left_index4";
	N[100]="left_middle1"; N[101]="left_middle2"; N[102]="left_middle3"; N[103]="left_middle4";
	N[104]="left_ring1"; N[105]="left_ring2"; N[106]="left_ring3"; N[107]="left_ring4";
	N[108]="left_pinky1"; N[109]="left_pinky2"; N[110]="left_pinky3"; N[111]="left_pinky4";
	
	N[112]="right_wrist"; N[113]="right_thumb1"; N[114]="right_thumb2"; N[115]="right_thumb3"; N[116]="right_thumb4";
	N[117]="right_index1"; N[118]="right_index2"; N[119]="right_index3"; N[120]="right_index4";
	N[121]="right_middle1"; N[122]="right_middle2"; N[123]="right_middle3"; N[124]="right_middle4";
	N[125]="right_ring1"; N[126]="right_ring2"; N[127]="right_ring3"; N[128]="right_ring4";
	N[129]="right_pinky1"; N[130]="right_pinky2"; N[131]="right_pinky3"; N[132]="right_pinky4";
	
	// name->idx
	for (int i=0;i<133;++i) S.name_to_idx[N[i]] = i;
	
	auto L = [&](int a,int b){ S.links.emplace_back(a,b); };
	// groups
	auto& G = S.linkGroups;
	auto& bodyG = G["body"];
	bodyG.color = ofColor::cyan;
	auto addB = [&](int a,int b){ bodyG.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	// body links
	addB(5,7); addB(7,9); addB(6,8); addB(8,10);
	addB(11,13); addB(13,15); addB(12,14); addB(14,16);
	addB(5,6); addB(11,12); addB(5,11); addB(6,12);
	addB(0,1); addB(0,2); addB(1,3); addB(2,4);
	addB(5,0); addB(6,0);
	
	// feet
	auto& feetG = G["feet"];
	feetG.color = ofColor::cyan;
	auto addF = [&](int a,int b){ feetG.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	addF(15,19); addF(19,17); addF(19,18);
	addF(16,22); addF(22,20); addF(22,21);
	
	// left hand group
	auto& lhandG = G["left_hand"];
	lhandG.color = ofColor::green;
	auto addLH = [&](int a,int b){ lhandG.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	auto chain = [&](int w,int a,int b,int c,int d){
		addLH(w,a); addLH(a,b); addLH(b,c); addLH(c,d);
	};
	chain(91,92,93,94,95);
	chain(91,96,97,98,99);
	chain(91,100,101,102,103);
	chain(91,104,105,106,107);
	chain(91,108,109,110,111);
	
	// right hand group
	auto& rhandG = G["right_hand"];
	rhandG.color = ofColor::red;
	auto addRH = [&](int a,int b){ rhandG.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	auto chainR = [&](int w,int a,int b,int c,int d){
		addRH(w,a); addRH(a,b); addRH(b,c); addRH(c,d);
	};
	chainR(112,113,114,115,116);
	chainR(112,117,118,119,120);
	chainR(112,121,122,123,124);
	chainR(112,125,126,127,128);
	chainR(112,129,130,131,132);
	
	// face
	auto& faceG = G["face"];
	faceG.color = ofColor::lightPink;
	auto addFace = [&](int a,int b){ faceG.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	auto loop = [&](int a,int b, bool close=false){
		for (int i=a+1;i<=b;++i) addFace(i-1,i);
		if (close) addFace(b,a);
	};
	loop(23,39);        // jaw
	loop(40,44);        // right eyebrow
	loop(45,49);        // left eyebrow
	loop(50,58);        // nose line/base (open)
	loop(59,64,true);   // right eye
	loop(65,70,true);   // left eye
	loop(71,82,true);   // outer mouth
	loop(83,90,true);   // inner mouth
	
	return S;
}

// --------------------
// Hand21 (single hand)
// --------------------
// MediaPipe-like indexing:
// 0 wrist
// Thumb  1-4, Index  5-8, Middle  9-12, Ring 13-16, Pinky 17-20
// using indices based on MediaPipe.
// https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
ofxRtmSkeletonSpec ofxRtmSkeletonSpec::getHand21Spec() {
	ofxRtmSkeletonSpec S;
	S.name = "hand21";
	auto& N = S.names;
	N.resize(21);
	N[0]="wrist";
	N[1]="thumb_cmc"; N[2]="thumb_mcp"; N[3]="thumb_ip"; N[4]="thumb_tip";
	N[5]="index_finger_mcp"; N[6]="index_finger_pip"; N[7]="index_finger_dip"; N[8]="index_finger_tip";
	N[9]="middle_finger_mcp"; N[10]="middle_finger_pip"; N[11]="middle_finger_dip"; N[12]="middle_finger_tip";
	N[13]="ring_finger_mcp"; N[14]="ring_finger_pip"; N[15]="ring_finger_dip"; N[16]="ring_finger_tip";
	N[17]="pinky_mcp"; N[18]="pinky_pip"; N[19]="pinky_dip"; N[20]="pinky_tip";
	
	for (int i=0;i<21;++i) S.name_to_idx[N[i]] = i;
	
	auto& palmG = S.linkGroups["palm"];
	palmG.color = ofColor::cyan;
	auto& fingersG = S.linkGroups["fingers"];
	fingersG.color = ofColor::yellow;
	auto addP = [&](int a,int b){ palmG.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	auto addF = [&](int a,int b){ fingersG.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	
	// Palm spokes
	addP(0,1); addP(0,5); addP(0,9); addP(0,13); addP(0,17);
	// Palm cross links (optional but nice for stability)
	addP(5,9); addP(9,13); addP(13,17);
	
	// Fingers
	auto chain = [&](int a,int b,int c,int d){ addF(a,b); addF(b,c); addF(c,d); };
	chain(1,2,3,4);       // thumb
	chain(5,6,7,8);       // index
	chain(9,10,11,12);    // middle
	chain(13,14,15,16);   // ring
	chain(17,18,19,20);   // pinky
	
	return S;
}

// --------------------
// Halpe26 (body + feet)
// --------------------
// Default order below follows a common Halpe26 convention:
// COCO-17 body   (0..16)
// + feet 6       (17..22): L big/small/heel, R big/small/heel
// + head/neck/pelvis anchors (23..25): head_top, neck, pelvis
// If your model's order differs, just edit the `N[...]` block.

ofxRtmSkeletonSpec ofxRtmSkeletonSpec::getHalpe26Spec() {
	ofxRtmSkeletonSpec S;
	S.name = "halpe26";
	auto& N = S.names;
	N.resize(26);
	
	// ---- Names (EDIT HERE if your indices differ) ----
	// COCO body 17
	N[0]  = "nose";
	N[1]  = "left_eye";
	N[2]  = "right_eye";
	N[3]  = "left_ear";
	N[4]  = "right_ear";
	N[5]  = "left_shoulder";
	N[6]  = "right_shoulder";
	N[7]  = "left_elbow";
	N[8]  = "right_elbow";
	N[9]  = "left_wrist";
	N[10] = "right_wrist";
	N[11] = "left_hip";
	N[12] = "right_hip";
	N[13] = "left_knee";
	N[14] = "right_knee";
	N[15] = "left_ankle";
	N[16] = "right_ankle";
	
	// Feet 6 (COCO-Foot ordering)
	N[17] = "left_big_toe";
	N[18] = "left_small_toe";
	N[19] = "left_heel";
	N[20] = "right_big_toe";
	N[21] = "right_small_toe";
	N[22] = "right_heel";
	
	// Trunk/head anchors (3)
	N[23] = "head_top";   // aka "top_head"
	N[24] = "neck";
	N[25] = "pelvis";     // hip center (midpoint of L/R hip)
	
	// name -> index
	for (int i=0;i<26;++i) S.name_to_idx[N[i]] = i;
	
	// ---- Edge groups ----
	auto& body  = S.linkGroups["body"];
	body.color = ofColor::cyan;
	auto& head  = S.linkGroups["head_neck"];
	head.color = ofColor::magenta;
	auto& feet  = S.linkGroups["feet"];
	feet.color = ofColor::yellow;
	
	auto addB = [&](int a,int b){ body.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	auto addH = [&](int a,int b){ head.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	auto addF = [&](int a,int b){ feet.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	
	// Trunk “spine” links (pelvis->hips, pelvis->neck->head_top)
	addB(25,11); addB(25,12);      // pelvis to hips
	addH(25,24); addH(24,23);      // pelvis->neck->head_top
								   // Nose/face anchors to neck for stability (optional)
	addH(24,0); addH(0,1); addH(0,2); addH(1,3); addH(2,4);
	
	// Arms
	addB(5,7); addB(7,9);          // left shoulder->elbow->wrist
	addB(6,8); addB(8,10);         // right shoulder->elbow->wrist
	addB(5,6);                     // across shoulders
	addB(24,5); addB(24,6);        // neck to shoulders
	
	// Legs
	addB(11,13); addB(13,15);      // left hip->knee->ankle
	addB(12,14); addB(14,16);      // right hip->knee->ankle
	addB(11,12);                   // across hips
	addB(25,11); addB(25,12);      // (already added; keep if you want duplication prevention elsewhere)
	
	// Feet (ankle->heel->toes)
	addF(15,19); addF(19,17); addF(19,18);  // left
	addF(16,22); addF(22,20); addF(22,21);  // right
	
	return S;
}

// --------------------
// COCO17 (body 2D keypoints)
// --------------------
ofxRtmSkeletonSpec ofxRtmSkeletonSpec::getCoco17() {
	ofxRtmSkeletonSpec S;
	S.name = "coco17";
	auto& N = S.names;
	N.resize(17);
	
	N[0]  = "nose";
	N[1]  = "left_eye";   N[2]  = "right_eye";
	N[3]  = "left_ear";   N[4]  = "right_ear";
	N[5]  = "left_shoulder"; N[6]  = "right_shoulder";
	N[7]  = "left_elbow";    N[8]  = "right_elbow";
	N[9]  = "left_wrist";    N[10] = "right_wrist";
	N[11] = "left_hip";      N[12] = "right_hip";
	N[13] = "left_knee";     N[14] = "right_knee";
	N[15] = "left_ankle";    N[16] = "right_ankle";
	
	for (int i=0;i<17;++i) S.name_to_idx[N[i]] = i;
	
	// ---- Edge groups ----
	auto& body = S.linkGroups["body"];
	body.color = ofColor::cyan;
	auto& head = S.linkGroups["head"];
	head.color = ofColor::lightPink;
	
	auto addB = [&](int a,int b){ body.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	auto addH = [&](int a,int b){ head.links.emplace_back(a,b); S.links.emplace_back(a,b); };
	
	// Body trunk & limbs
	addB(5,7); addB(7,9);     // left arm
	addB(6,8); addB(8,10);    // right arm
	addB(11,13); addB(13,15); // left leg
	addB(12,14); addB(14,16); // right leg
	addB(5,6);                // across shoulders
	addB(11,12);              // across hips
	addB(5,11); addB(6,12);   // shoulders to hips (diagonals)
	
	// Head & neck area
	addH(5,0); addH(6,0);     // shoulders to nose
	addH(0,1); addH(0,2);     // nose to eyes
	addH(1,3); addH(2,4);     // eyes to ears
	
	return S;
}
