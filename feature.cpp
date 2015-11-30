#include <cmath>
#include <queue>
#include "feature.hpp"
#include "tree.hpp"

// Static variables
double RawFeature::complexity = 0;
double ProductFeature::complexity = 0;
double ThresholdFeature::complexity = 0;

// Returns sample expectation of the given feature.
double Feature::GetSampleExpectation() {
  return sample_expectation;
}

// Returns un-normalized population expectation of the given feature
// that is currently stored in the feature. It is either NAN or was
// computed on some space using ComputeUnnormalizedPopulationExpectation.
double Feature::GetUnnormalizedPopulationExpectation() {
  return population_expectation;
}

// Computes a sample expectation of the given feature
// and stores it internally.
void Feature::ComputeSampleExpectation(Sample &sample) {
  double sum = 0.0;
  int count = 0;
  for (auto e : sample) {
    count++;
    sum += FeatureMap(e);
  }
  sample_expectation = sum / count;
}

// Returns un-normalized expectation of the given feature
// wrt the weights of each point in the provided space.
// To get expectation one needs to further divide the result
// by the sum of weights of all the points in the space.
void Feature::ComputeUnnormalizedPopulationExpectation(Space &space) {
  double expectation = 0.0;
  for (auto &p : space) {
    expectation += p.GetProbWeight() * FeatureMap(&p);
  }
  population_expectation =  expectation;
}

// Constructor for a raw feature. Client needs to specify
// index of the raw feature that is being constructed.
RawFeature::RawFeature(int i){
  index = i;
  sample_expectation = NAN;
  population_expectation = NAN;
}

// Returns the value of this raw feature at specified point.
double RawFeature::FeatureMap(Point *point) {
  return point->GetRawFeature(index);
}

// Returns complexity of the class of raw features.
double RawFeature::Complexity() {
  return complexity;
}

// Sets complexity of this class of features.
// Note that complexity may often depend on a specific data set used.
void RawFeature::SetComplexity(double value) {
  complexity = value;
}

// Constructor for product feature. Client needs to specify
// indices of raw features that are to be multiplied.
ProductFeature::ProductFeature(int i, int j){
  first_index = i;
  second_index = j;
  sample_expectation = NAN;
  population_expectation = NAN;
}

// Returns the value of the product feature at specified point.
double ProductFeature::FeatureMap(Point *point) {
  return (point->GetRawFeature(first_index) *
	  point->GetRawFeature(second_index));
}

// Returns complexity of the class of prodcut features.
double ProductFeature::Complexity() {
  return complexity;
}

// Sets complexity of this class of features.
// Note that complexity may often depend on a specific data set used.
void ProductFeature::SetComplexity(double value) {
  complexity = value;
}

// Constructor for threshold feature. Client needs to specify
// index of raw feature and threshold value.
ThresholdFeature::ThresholdFeature(int i, double theta) {
  index = i;
  threshold = theta;
  sample_expectation = NAN;
  population_expectation = NAN;
}

// Returns the value of the threshold feature at specified point.
// The value is 1 if the specified raw feature is above threshold
// and the value is 0 otherwise. According to this definition,
// if raw feature is missing (i.e. is NAN) then the value is also 0.
double ThresholdFeature::FeatureMap(Point *point) {
  return ((point->GetRawFeature(index) > threshold) ? 1 : 0);
}

// Returns complexity of the class of threshold features.
double ThresholdFeature::Complexity() {
  return complexity;
}

// Sets complexity of this class of features.
// Note that complexity may often depend on a specific data set used.
void ThresholdFeature::SetComplexity(double value) {
  complexity = value;
}

// Constructor for tree feature. Client needs to specify
// the tree that defines this feature.
TreeFeature::TreeFeature(Node *node) {
  root = node;
  sample_expectation = NAN;
  population_expectation = NAN;
}

// Destructor for tree feature. Deletes the tree contained in this feature.
TreeFeature::~TreeFeature() {
  std::queue<Node*> q;
  q.push(root);
  while (!q.empty()) {
    Node *node = q.front();
    q.pop();
    if (node->GetLeftChild()) {
      q.push(node->GetLeftChild());
    }
    if (node->GetRightChild()) {
      q.push(node->GetRightChild());
    }
    delete node;
  }
}


// Returns the value of the tree feature map at the specified point.
// The value is defined by the underlying tree itself.
double TreeFeature::FeatureMap(Point *point) {
  Node *node = root;
  while (!(node->IsLeaf())) {
    node = node->Child(point);
  }
  return node->GetValue();
}

// Returns complexity of the tree feature.
double TreeFeature::Complexity() {
  return complexity;
}

// Sets complexity of this class of features.
// Note that complexity may often depend on a specific data set used.
void TreeFeature::SetComplexity(double value) {
  complexity = value;
}

// Sets (unnormalized) population and sample expectations of this
// tree feature to value based on population weights, sample counts
// and values stored in its leafs.
void TreeFeature::ComputeTreeExpectations() {
  double up_expectation = 0.0;
  double us_expectation = 0.0;
  double sample_count = 0.0;
  std::queue<Node*> q;
  q.push(root);
  while (!q.empty()) {
    Node *node = q.front();
    q.pop();
    if (node->IsLeaf()) {
      up_expectation += node->GetValue() * node->GetPopulationWeight();
      us_expectation += node->GetValue() * node->GetSampleCount();
      sample_count += node->GetSampleCount();
    } else {
      q.push(node->GetLeftChild());
      q.push(node->GetRightChild());
    }
  }
  population_expectation = up_expectation;
  sample_expectation = us_expectation / sample_count;
}

// Returns the size (number of nodes) of the given tree.
int TreeFeature::TreeSize() {
  std::queue<Node*> q;
  q.push(root);
  int tree_size = 0;
  while (!q.empty()) {
    Node *node = q.front();
    q.pop();
    tree_size++;
    if (node->GetLeftChild()) {
      q.push(node->GetLeftChild());
    }
    if (node->GetRightChild()) {
      q.push(node->GetRightChild());
    }
  }
  return tree_size;
}

// Constructor for monomial feature. Client needs to specify
// the powers for each raw feature.
MonomialFeature::MonomialFeature(std::vector<int> &pwrs) {
  powers = pwrs;
  sample_expectation = NAN;
  population_expectation = NAN;
}

// Returns the value of the monomial feature map at the specified point.
double MonomialFeature::FeatureMap(Point *point) {
  double result = 1.0;
  for (unsigned index = 0; index < powers.size(); index++) {
    result *= std::pow(point->GetRawFeature(index), powers[index]);
  }
  return result;
}

// Returns complexity of the monomial feature.
double MonomialFeature::Complexity() {
  return complexity;
}

// Sets complexity of this class of features.
// Note that complexity may often depend on a specific data set used.
void MonomialFeature::SetComplexity(double value) {
  complexity = value;
}

// Sets monomial expectations to sepcified values.
void MonomialFeature::MonomialExpectations(double population, double sample) {
  population_expectation = population;
  sample_expectation = sample;
}

// Returns the power of this monomial feature.
int MonomialFeature::GetPower() {
  int sum = 0;
  for (int p : powers) {
    sum += p;
  }
  return sum;
}
