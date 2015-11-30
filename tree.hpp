#include "space.hpp"

#ifndef TREE_HPP
#define TREE_HPP

// This is a class that represents a node in a decision tree.
// If node is a leaf it will contain points in the space and samples
// that correspond to this leaf, as well as the weight of all points
// contained in it and the value associated with it.
// If node is an internal node then it will contain a binary question
// (threshold, feature), as well as both left and right child.
// See tree_test.cpp and tree.cpp for sample usage.
class Node {
public:
  double GetValue();
  double GetPopulationWeight();
  int GetSampleCount();
  Node *GetLeftChild();
  Node *GetRightChild();
  void SetThreshold(double threshold);
  void SetFeature(int feature);
  void SetLeftChild(Node *child);
  void SetRightChild(Node *child);
  void SetValue(double value);
  std::vector<Point*>::iterator PointsBegin();
  std::vector<Point*>::iterator PointsEnd();
  std::vector<Point*>::iterator SamplesBegin();
  std::vector<Point*>::iterator SamplesEnd();
  void ClearPoints();
  void ClearSamples();
  void AddPoint(Point *point);
  void AddSample(Point *point);
  bool IsLeaf();
  Node *Child(Point *point);
private:
  int feature;
  double threshold;
  double value;
  Node *left_child;
  Node *right_child;
  double weight;
  std::vector<Point*> points;
  std::vector<Point*> samples;
};

#endif
