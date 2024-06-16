#!/bin/bash
g++ -O0 -g -o runBIED BIED.cpp PathManager.cpp utils.cpp EdgeDetector.cpp EdgeDetectorHyperparams.cpp -lstdc++fs `pkg-config opencv4 --cflags --libs`
