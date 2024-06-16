#include "LinePattern.h"

LocalLine::LocalLine()
{
    for(int idx = 0; idx  < this->numPattern; idx++)
    {
        SearchRegion leftSR = SearchRegion(this->length, this->width, 360 - idx * 180 / this->numPattern);
        leftLinkMasks.push_back(leftSR.pDiff);
        SearchRegion rightSR = SearchRegion(this->length, this->width, 180 - idx * 180 / this->numPattern);
        rightLinkMasks.push_back(rightSR.pDiff);
    }
}
