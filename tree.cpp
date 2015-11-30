#include "tree.hpp"
#include <cstddef>

// Returns the value of this node.
double Node::GetValue() {
  return value;
}

// Returns the weight of points stored at this node.
double Node::GetPopulationWeight() {
  return weight;
}

// Returns the number of sample points stored at this node.
int Node::GetSampleCount() {
  return samples.size();
}

// Returns the left child of this node (possibly NULL).
Node *Node::GetLeftChild() {
  return left_child;
}

// Returns the right child of this node (possibly NULL).
Node *Node::GetRightChild() {
  return right_child;
}

// Sets threshold for this node to the given value.
void Node::SetThreshold(double val) {
  threshold = val;
}

// Sets feature for this node to the given value.
void Node::SetFeature(int index) {
  feature = index;
}

// Sets left child of this node.
void Node::SetLeftChild(Node *child) {
  left_child = child;
}

// Sets right child of this node.
void Node::SetRightChild(Node *child) {
  right_child = child;
}

// Sets value of this node.
void Node::SetValue(double val) {
  value = val;
}

// Removes all the points stored in this node.
void Node::ClearPoints() {
  //points.clear();
  std::vector<Point*>().swap(points);
  weight = 0.0;
}

// Removes all samples stored in this node.
void Node::ClearSamples() {
  std::vector<Point*>().swap(samples);
  //samples.clear();
}

// Adds a point to this node.
void Node::AddPoint(Point *point) {
  points.push_back(point);
  weight += point->GetProbWeight();
}

// Adds a sample to this node.
void Node::AddSample(Point *point) {
  samples.push_back(point);
}

// Returns true iff this node is a leaf (i.e. both children are NULL).
bool Node::IsLeaf() {
  return ((left_child == NULL) && (right_child == NULL));
}

// Returns the child of this node that contains given point.
// NULL is returned if this node is a leaf.
Node *Node::Child(Point *point) {
  if (point->GetRawFeature(feature) < threshold) {
    return left_child;
  }
  return right_child;
}

// Returns an iterator pointing to the first point in this node.
std::vector<Point*>::iterator Node::PointsBegin() {
  return points.begin();
}

// Returns an iterator pointing to the point after the last point in this node.
std::vector<Point*>::iterator Node::PointsEnd() {
  return points.end();
}

// Returns an iterator pointing to the first sample in this node.
std::vector<Point*>::iterator Node::SamplesBegin() {
  return samples.begin();
}

// Returns an iterator pointing to the sample after the last sample
// in this node.
std::vector<Point*>::iterator Node::SamplesEnd() {
  return samples.end();
}
