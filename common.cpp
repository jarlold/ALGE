/* This is for stuff that other libraries would have
   that I don't. Maybe I should have called it utils..
*/

#pragma once
#include <vector>
#include <stdio.h>
#include <memory>
#include <numeric>
#include <windows.h>
#include <psapi.h>
#include <iostream>

#include "node_calculus.cpp"
#include "flags.cpp"

std::vector<NumKin> range(int length) {
    std::vector<NumKin> nums(length);
    for (int i=0; i<length; i++) {
        nums[i] = i; 
    }
    return nums;
}

#ifdef IS_ARDUINO
float randomFloat(float min = -1.0f, float max = 1.0f) {
  // Generate random long in [0, 1000000]
  int randInt = random(1000000);
  
  // Convert to float in [0.0, 1.0)
  float randNorm = randInt / 1000000.0f;
  
  // Scale to [min, max)
  return min + randNorm * (max - min);
}
#endif

#ifndef IS_ARDUINO
float randomFloat(float min = -1.0f, float max = 1.0f) {
	// Generate random long in [0, 1000000]
	int randInt = rand() % 1000000;
	
	// Convert to float in [0.0, 1.0)
	float randNorm = randInt / 1000000.0f;
	
	// Scale to [min, max)
	return min + randNorm * (max - min);
}

int PrintPeakMemoryUsage() {
	PROCESS_MEMORY_COUNTERS_EX pmc;
	if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
	{
		SIZE_T peakWorkingSet = pmc.PeakWorkingSetSize;
		SIZE_T peakPagefileUsage = pmc.PeakPagefileUsage;  // aka peak commit
		SIZE_T privateBytes = pmc.PrivateUsage;
		
		std::cout << "Peak Working Set     : " << peakWorkingSet  << " B\n";
		std::cout << "Peak Commit (Private): " << peakPagefileUsage  << " B\n";
		std::cout << "Current Private Bytes: " << privateBytes  << " B\n";
		
		return peakWorkingSet;
	}
	return -1;
}

#endif

/*
NumKin mean(const std::vector<NumKin>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
}

NumKin stddev(const std::vector<NumKin>& v, NumKin m) {
    NumKin sum = 0.0f;
    for (auto val : v) sum += (val - m) * (val - m);
    return std::sqrt(sum / v.size());
}


std::vector<NumKin> normalizeDataset(std::vector<NumKin>& vec, NumKin new_min, NumKin new_max) {
    NumKin min_val = *std::min_element(vec.begin(), vec.end());
    NumKin max_val = *std::max_element(vec.begin(), vec.end());

    // Avoid division by zero
    if (max_val - min_val == 0) {
        std::vector<NumKin> normalized(vec.size());
        std::fill(normalized.begin(), normalized.end(), (new_min+new_max)/2.0);  // All elements are the same
        return normalized;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = new_min + (vec[i] - min_val) * (new_max - new_min) / (max_val - min_val);
    }

    return vec;
}
*/

