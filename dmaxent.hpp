#include "space.hpp"
#include "feature.hpp"
#include "wlearner.hpp"

#ifndef DMAXENT_HPP
#define DMAXENT_HPP

// This class represents a (Deep) Max Entropy model.
//
// This class offers the following main methods:
//   DMaxEntModel(params) - initializes model with given parameters and data;
//                          see below for more details on arguments
//   Fit() - fits the model initialized by constructor
//   LogLoss(sample) - evaluates fitted model using provided test sample;
//                      the performance metric that is used is log loss
//   AUC(sample) - evaluates fitted model using provided test sample;
//                      the performance metric that is used is AUC
//   ~DMaxEntModel() - destructor
//
// This class also provides auxiliary methods listed below (primarily for
// testing purposes):
//   GetDescentDirection() - returns the last direction (feature) used
//                           when optimizing using Fit()
//   GetStepSize() - returns the last step size used in Fit()
//   GetWeight(coordinate) - returns the weight of the specified coordinate
//                           in the model
//   GetNormalizer() - returns value of normalizer stored in the model
//
//
// Recall that (Deep) Max Entropy model is a Gibbs distribution
// over some space with weighted linear combination of features in exponent.
// Therefore, this class consists of underlying space
// (over which we are trying to fit density) and
// a weighted vector of features that are used in this model.
// To fit a model we also need two regularization parameters,
// uniform bound on features (lambda) and a maximum number of iterations
// for optimization procedure. Thus to initialize a model using constructor
// one needs to specify:
//   regularization parameters (alpha and beta)
//   maximum number of steps for opitmization procedure
//   version of step size formula (1 or 2) 
//   uniform bound on feature values
//   space
//   sample
//   features used by the model
//   weak learners used by the model
//
// Sample usage:
// DMaxEntModel *model = new DMaxEntModel(alpha, beta,  max_steps, ver, lambda,
//                                        space, sample, features, learners);
// model->Fit();
// double loss = model->Evaluate(sample);
// delete model;
class DMaxEntModel{
public:
  DMaxEntModel(double model_parameter_alpha, double model_parameter_beta,
	       double max_steps, int version, double lambda,
	       bool stop_if_converged, Space *space,
	       Sample sample, std::vector<Feature*> *features,
	       std::vector<WLearner*> weak_learners, Sample test_sample);
  ~DMaxEntModel();
  void Fit();
  double LogLoss(Sample *sample);
  double AUC(Sample *sample);
  int GetDescentDirection();
  double GetStepSize();
  double GetWeight(int coordinate);
  double GetNormalizer();
  typedef std::vector< std::pair<double, Feature*> >::iterator FeatureIterator;
  FeatureIterator FeatureBegin();
  FeatureIterator FeatureEnd();
private:
  void FindDescentDirection();
  void FindStepSize1();
  void FindStepSize2();
  void UpdateModel();
  std::vector<std::pair<double, Feature*>> weighted_features;
  std::vector<WLearner*> weak_learners; 
  Space *space;
  Sample sample;
  double normalizer;
  double model_parameter_alpha;
  double model_parameter_beta;
  double lambda;
  double max_descent_steps;
  double step_size;
  double model_gradient;
  int direction;
  int version;
  bool stop_if_converged;
  Sample test_sample;
};

#endif
