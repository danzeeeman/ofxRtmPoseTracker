

#include "ofxRtmModelLoader.h"
#include "ofJson.h"
#include "ofLog.h"

namespace {
std::string toDataRelativePath(const std::string& absolutePath) {
	std::string dataPath = ofToDataPath("", true); // absolute data/
	
	// Ensure trailing slash
	if (!dataPath.empty() && dataPath.back() != '/') {
		dataPath += "/";
	}
	
	// Absolute paths returned by OF already use '/'
	if (absolutePath.rfind(dataPath, 0) == 0) {
		return absolutePath.substr(dataPath.size());
	}
	
	// Not inside data folder → return unchanged
	return absolutePath;
}
}

//--------------------------------------------------------------
std::string ofxRtmModelLoader::loadModel(const std::string& modelsRoot, const std::string& modelUrl ) {
	return loadModel( "", modelsRoot, modelUrl );
}

//--------------------------------------------------------------
std::string ofxRtmModelLoader::loadModel(const std::string& aTag, const std::string& modelsRoot, const std::string& modelUrl ) {
	ofHttpRequest req;
	req.timeoutSeconds = 60 * 60; // an hour
	return loadModel(aTag, modelsRoot, modelUrl, req );
}

//--------------------------------------------------------------
std::string ofxRtmModelLoader::loadModel(const std::string& modelsRoot, const std::string& modelUrl, ofHttpRequest& arequest ) {
	return loadModel("", modelsRoot, modelUrl, arequest);
}

//--------------------------------------------------------------
std::string ofxRtmModelLoader::loadModel(const std::string& aTag, const std::string& modelsRoot, const std::string& modelUrl, ofHttpRequest& arequest ) {
	std::string modelsDir = ofToDataPath(modelsRoot, true);
	
	if( !ofDirectory::doesDirectoryExist(modelsDir)) {
		ofDirectory(modelsDir).create(true);
	}
	
	// ---- load registry ----
	ofJson registry;
	std::string registryPath = ofToDataPath(modelsRoot + "/.registry.json", true);
	
	if (ofFile::doesFileExist(registryPath)) {
		registry = ofLoadJson(registryPath);
	}
	
	std::string id = hashString(modelUrl);
	
	std::string tag = aTag;
	if( tag.empty() ) {
		tag = id;
	}
	
	// ---- already installed ----
	if (registry.contains(tag)) {
		std::string path = registry[tag]["path"];
		ofLogNotice("ofxRtmModelLoader::loadModel") << " found: " << tag << " - model in registry: " << path;
		if( ofFile::doesFileExist(path)) {
			return path;
		}
	}
	
	ofLogNotice("ofxRtmModelLoader::loadModel") << "Installing model from " << modelUrl;
	
	// ---- download ----
	std::string zipPath = ofToDataPath(modelsRoot + "/" + id + ".zip", true);
	
	ofURLFileLoader floader;
	ofHttpRequest req = arequest;
	req.url = modelUrl;
	req.method = ofHttpRequest::GET;
	ofHttpResponse res = floader.handleRequest(req);
	
	if (res.status != 200) {
		ofLogError("ofxRtmModelLoader::loadModel") << "Download failed";
		return "";
	}
	
	if(!ofBufferToFile(zipPath, res.data, true)) {
		ofLogError("ofxRtmModelLoader::loadModel") << "Buffer to file Failed at" << zipPath;
		return "";
	}
	
	std::string modelDir = ofToDataPath(ofFilePath::addTrailingSlash(modelsDir)+"m"+id, true);
	
	// ---- unzip ----
	if (!unzipTo(zipPath, modelDir)) {
		ofLogError("ofxRtmModelLoader::loadModel") << "Unzip failed";
		return "";
	}
	
	// ---- locate model dir ----
	std::string modelPath = findPathContainingExtRecursive(modelDir, "onnx", 0, 8);
	
	if (modelPath.empty()) {
		ofLogError("ofxRtmModelLoader::loadModel") << "Model not found after unzip";
		return "";
	}
	
	// ---- update registry ----
	registry[tag] = {
		{"url", modelUrl},
		{"path", toDataRelativePath(modelPath)}
	};
	
	ofSavePrettyJson(registryPath, registry);
	
	// ---- cleanup ----
	if( ofFile::doesFileExist(zipPath)) {
		ofFile::removeFile(zipPath);
	}
	
	return modelPath;
}

//--------------------------------------------------------------
bool ofxRtmModelLoader::unzipTo(const std::string& zipPath, const std::string& dstDir) {
#if defined(TARGET_OSX) || defined(TARGET_LINUX)
	std::string cmd = "unzip -o \"" + zipPath + "\" -d \"" + dstDir + "\"";
	return system(cmd.c_str()) == 0;
#else
	std::string cmd = "powershell -Command \"Expand-Archive -Force '" + zipPath + "' '" + dstDir + "'\"";
	return system(cmd.c_str()) == 0;
#endif
}

//--------------------------------------------------------------
std::string ofxRtmModelLoader::findPathContainingExtRecursive( const std::string& root, const std::string& ext, int depth, int maxDepth) {
	ofDirectory dir(root);
	dir.listDir();
	
	// 1. check files in this directory
	for (auto& f : dir) {
		if (f.isFile() && ofToLower(f.getExtension()) == ofToLower(ext)) {
			return ofToDataPath(f, true);
		}
	}
	
	// 2. recurse into subdirectories
	for (auto& f : dir) {
		if (f.isDirectory()) {
			std::string result = findPathContainingExtRecursive(ofToDataPath(f, true), ext, depth + 1, maxDepth);
			if (!result.empty()) {
				return result;
			}
		}
	}
	
	return "";
}

//--------------------------------------------------------------
std::string ofxRtmModelLoader::hashString(const std::string& s) {
	std::size_t h = std::hash<std::string>{}(s);
	std::stringstream ss;
	ss << std::hex << h;
	return ss.str();
}
