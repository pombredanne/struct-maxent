#include <cmath>
#include "space.hpp"

// Constructor for an instance of class space.
Space::Space() {
  finalized = false;
}

// Adds specified point to the space. Returns key of the point
// on success and -1 if this space has already been finalized.
int Space::AddPoint(Point &point) {
  if (finalized) {
    return -1;
  }
  points.push_back(point);
  return (points.size() - 1);
}

// Returns a reference to a point with a specified key in the space.
// If key does not match any point in the space out_of_range exception
// is thrown.
Point& Space::GetPoint(int key) {
  return points.at(key);
}

// Finalizes this space. Once space is finalized no points can be added to
// it and all points in the space are finalized themselves.
void Space::Finalize() {
  for (auto &point : points) {
    point.Finalize();
  }
  finalized = true;
}

// Returns number of points in this space.
int Space::NumPoints() {
  return points.size();
}

// Returns iterator to the beginning of the space container.
Space::SpaceIterator Space::begin() {
  return points.begin();
}

// Returns iterator to the end of the space container.
Space::SpaceIterator Space::end() {
  return points.end();
}

// Constructor for a point with a specified id.
Point::Point(int point_id) {
  id = point_id;
  finalized = false;
  probability_weight = 1.0;
}

// Returns probabilistic weight of this point.
double Point::GetProbWeight() {
  return probability_weight;
}

// Returns specified raw feature value of this point.
double Point::GetRawFeature(int index) {
  return (index < raw_features.size() ? raw_features[index] : NAN);
}

// Sets probabilistic weight of this point with specified value.
void Point::SetProbWeight(double value) {
  probability_weight = value;
}

// Adds a raw feature value to this point. If point has been finalized
// returns -1 otherwise index of the feature is returned.
int Point::AddRawFeature(double value) {
  if (finalized) {
    return -1;
  }
  raw_features.push_back(value);
  return (raw_features.size() - 1);
}

// Finalizes this point. Once point is finalized, its raw features can not
// be modified.
void Point::Finalize() {
  finalized = true;
}

// Returns id of this point.
int Point::GetId() {
  return id;
}


// Returns number of raw features for this point.
int Point::NumRawFeatures() {
  return raw_features.size();
}
