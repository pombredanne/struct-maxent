#include <string>
#include "space.hpp"
#include "tree.hpp"

#ifndef FEATURE_HPP
#define FEATURE_HPP

// This is an abstract class that represents a generic feature
// (a map from an input space to real numbers).
class Feature{
public:
  double GetSampleExpectation();
  double GetUnnormalizedPopulationExpectation();
  void ComputeSampleExpectation(Sample &sample);
  void ComputeUnnormalizedPopulationExpectation(Space &space);
  virtual void SetComplexity(double value) = 0;
  virtual double Complexity() = 0;
  virtual double FeatureMap(Point *point) = 0;
protected:
  double sample_expectation;  // expectation wrt observed sample
  double population_expectation; // expectation wrt population density
};

// This class represents a raw feature, that is a map
// from input space to real numbers that is equal to one of the
// raw features
class RawFeature: public Feature{
public:
  RawFeature(int i);
  double FeatureMap(Point* point); // override
  double Complexity(); // override
  void SetComplexity(double value); // override
private:
  int index;  // index of the raw feature in the feature vector
  static double complexity; // complexity of this feature class
};

// This class represents a product feature, that is a map
// from input space to real numbers that is equal to a product of the
// two raw features. Note that this includes squared features as well.
class ProductFeature: public Feature{
public:
  ProductFeature(int i, int j);
  double FeatureMap(Point* point); // override
  double Complexity(); // override
  void SetComplexity(double value); // override
private:
  int first_index;   // index of the first raw feature
  int second_index;  // index of the second raw feature
  static double complexity; // complexity of this feature class
};

// This class represents a threshold feature, that is a map
// from input space to real numbers that is equal to 1 if specified
// raw feature above given threshold and 0 otherwise. According
// to this definition, if raw feature is missing (i.e. is NAN)
// then the value is also 0.
class ThresholdFeature: public Feature{
public:
  ThresholdFeature(int i, double theta); 
  double FeatureMap(Point* point); // override
  double Complexity(); // override
  void SetComplexity(double value); // override
private:
  int index;  // index of the raw feature in the feature vector
  double threshold;
  static double complexity; // complexity of this feature class
};

// This class represents a tree feature, that is a map from input space
// to real numbers. Each tree feature corresponds to a particular partition
// of space and the value of the feature map on each partition is
// the value at the corresponding leaf of the tree.
class TreeFeature: public Feature{
public:
  TreeFeature(Node* node);
  ~TreeFeature();
  double FeatureMap(Point* point); // override
  double Complexity(); // override
  void SetComplexity(double value); // override
  void ComputeTreeExpectations();
  int TreeSize();
private:
  Node* root; // tree that defines this feature
  double complexity; // complexity of this particular tree feature
};

class MonomialFeature: public Feature{
public:
  MonomialFeature(std::vector<int> &powers);
  double FeatureMap(Point* point); // override
  double Complexity(); // override
  void SetComplexity(double value); // override
  void MonomialExpectations(double population_expectation,
			    double sample_expectation);
  int GetPower();
private:
  std::vector<int> powers; // powers for each raw feature
  double complexity; // complexity of this monomial feature
};

#endif
