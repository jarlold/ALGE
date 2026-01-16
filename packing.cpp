/* This file is devoted to "memory packing tricks" and the like.
   Essentially, here is the weird messy compression techniques that
   let us run such big models on such tiny hardware.
*/
#pragma once
#include "multidimensional.cpp"
#include "common.cpp"
#include <vector>
#include <list>

struct PackedVector {
	float base;
	float increment;
	std::vector<short> offsets;
};

PackedVector packVector(ValueVector& v) {
	float base = 0;	
	float min;
	float max;
	
	for (float f : v) { 
		base += f;
		if (!min || f < min) min = f;
		if (!max || f > max) max = f;
	}
	
	base /= (float) v.size();
	float r = (max - min) / (255.0*2);

	// Then we'll store the difference between
	// the base and the real value as an integer.
	std::vector<short> offsets(v.size());
	for (int i=0; i < (int) v.size(); i++) {
		offsets[i] = (v[i] - base) / r;
	}
	
	PackedVector p = { base, r, offsets };
	
	return p;
}

ValueVector unpackVector(PackedVector& p) {
	ValueVector v(p.offsets.size());
	
	for (int i=0; i < (int)p.offsets.size(); i++) {
		v[i] = ((float) p.offsets[i])*p.increment + p.base;
	}
	
	return v;
}

void saveForwardsAsVector(ValueVector& values, const NodePtr& root) {
	std::list<NodePtr> visits;
	listOfNodes(root, visits);
	values = ValueVector(visits.size());
	int inc = 0;
	for (NodePtr& p : visits) {
		values[inc] = p->value;
		p->visited = false;
		inc++;
	}
}

void loadForwardsFromVector(ValueVector& values, NodePtr& root) {
	std::list<NodePtr> visits;
	listOfNodes(root, visits);
	int inc = 0;
	for (NodePtr& p : visits) {
		p->value = values[inc];
		p->visited = false;
		inc++;
	}
}

/* For testing various forms of packing.
int main() {
	ValueVector v(100);
	for (int i =0; i < 100; i++) {
		v[i] = (float) i / 100 * i + 22 - pow(i, 0.4) + (i%3);
	}
	
	PackedVector p = packVector(v);
	ValueVector unpacked = unpackVector(p);
	
	printf("%d * %f + %f = %f, %f\r\n", p.offsets[3], p.increment, p.base, unpacked[3], v[3]);
	printf("%llu ", sizeof(v[0]) * v.size());
	printf("%llu \r\n", sizeof(p.offsets[0]) * p.offsets.size());

	for (int i =0; i < 100; i++) {
		printf("%f, %d\r\n", p.base, p.offsets[i]);
		printf("%f %f %f\r\n", v[i], unpacked[i], v[i] - unpacked[i]);
	}
}

*/
