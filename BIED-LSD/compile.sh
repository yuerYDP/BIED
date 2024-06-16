#!/bin/bash
g++ -O0 -g -o runexe BIED-LSD.cpp PathManager.cpp utils.cpp EdgeDetector.cpp EdgeDetectorHyperparams.cpp EdgeSegment.cpp LinePattern.cpp LineSegment.cpp SearchRegion.cpp LineDetector.cpp -lstdc++fs `pkg-config opencv4 --cflags --libs`
#g++ -O0 -g -o runBIED BIED.cpp PathManager.cpp utils.cpp EdgeDetector.cpp EdgeDetectorHyperparams.cpp -lstdc++fs `pkg-config opencv4 --cflags --libs`
