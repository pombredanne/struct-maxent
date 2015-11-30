#include <map>
#include "tree.hpp"
#include "space.hpp"
#include "feature.hpp"

#ifndef WLEARNER_HPP
#define WLEARNER_HPP

// This is an abstract class that represents a generic weak learner.
// The main purpose of weak learners is to train feature maps.
class WLearner{
public:
  virtual void Train(Space &space, Sample &sample,
		     Feature **feature, double *gradient) = 0;
};

// This class represents a tree weak learner. Given a sample over
// an underlying space this class can be trained to return a tree
// feature map.
// Sample usage:
//  TreeLearner *tlearner = new TreeLearner(num_features, alpha, beta, vtot);
//  Feature* tree_feature = tlearner->Train(space, sample);
// Here num_features is the number of raw features, alpha and beta
// are regularization parameters for structural maxent model and
// vtot is a vector of maps from feature values to thresholds.
// Each value is mapped to the next largest threshold for this feature.
// This class also provides a number of auxillilary methods used in
// training. For further details consult wlearner.cpp.
class TreeLearner : public WLearner{
public:
  TreeLearner(int num_features, double model_parameter_alpha,
	      double model_parameter_beta,
	      std::vector< std::map<double, double> > value_to_thresholds);
  void Train(Space &space, Sample &sample,
	     Feature **feature, double * gradient); // override
  void BestThreshold(int feature, Node *node, double old_expectation_diff,
		     double normalizer, int sample_size, int tree_size,
		     double *threshold, double *gradient, double *left_value,
		     double *new_expectation_diff);
  double Gradient(int tree_size, int sample_size, double expectation_diff);
  double TreeComplexity(int tree_size, int sample_size);
  void GrowTree(Node *node, double threshold, int feature_index, int left_val,
		Node **left_child, Node **right_child);
  std::map<double, std::pair<double, int>>
    BuildThresholdToWeightsMap(Node *node, int index);
private:
  int num_features;
  double model_parameter_alpha;
  double model_parameter_beta;
  std::vector< std::map<double, double> > value_to_thresholds;
};

class MonomialLearner : public WLearner{
public:
  MonomialLearner(int num_features, double model_parameter_alpha,
		   double model_parameter_beta, double feature_bound);
  void Train(Space &space, Sample &sample,
	     Feature **feature, double * gradient); // override
  double Gradient(int power, int sample_size, double difference);
  double MonomialComplexity(int power, int sample_size);
  void BestFeature(const std::vector<double> &point_values,
		   const std::vector<double> &sample_values,
		   Space &space, Sample &sample, double normalizer, int power,
		   double *candidate_gradient, int *candidate_feature,
		   double *candidate_population_expectation,
		   double *candidate_sample_expectation);
private:
  int num_features;
  double model_parameter_alpha;
  double model_parameter_beta;
  double feature_bound;
};

#endif 
