#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include "dmaxent.hpp"
#include "constants.hpp"
#include "glog/logging.h"


// Returns 1 if x > 0 and -1 otherwise.
inline double sgn(double x) {
  return (x > 0 ? 1.0 : -1.0);
}

// Sets direction attribute of this model to an index of a feature (stored 
// internally) that corresponds to coordinate descent direction based
// on the current state of the model (weights of features and probability
// density over the space).
// Note that it is assumed that sample expectations of features has been
// already computed.
void DMaxEntModel::FindDescentDirection() {
  Feature *feature;
  int best_feature_index = 0;
  double beta;
  double gradient;
  double diff_expectations;
  double feature_weight;
  double best_absolute_gradient = -1.0;
  for (unsigned index = 0; index < weighted_features.size(); index++) {
    feature_weight = weighted_features[index].first;
    feature = weighted_features[index].second;
    feature->ComputeUnnormalizedPopulationExpectation(*space);
    diff_expectations = -feature->GetSampleExpectation() + 
      (feature->GetUnnormalizedPopulationExpectation() / normalizer);
    beta = 2 * model_parameter_alpha * feature->Complexity() +
      model_parameter_beta;
    if (std::abs(feature_weight) > gTolerance) {
      gradient = beta * sgn(feature_weight) + diff_expectations;
    } else if (std::abs(diff_expectations) <  beta) {
      gradient = 0;
    } else {
      gradient = -beta * sgn(diff_expectations) + diff_expectations;
    }
    if (std::abs(gradient) >= best_absolute_gradient) {
      best_feature_index = index;
      best_absolute_gradient = std::abs(gradient);
    }
  }
  Feature *new_feature;
  bool new_feature_found = false;
  for (auto learner : weak_learners) {
    double gradient;
    Feature *feature;
    learner->Train(*space, sample, &feature, &gradient);
    if (std::abs(gradient) > best_absolute_gradient + gTolerance) {
      new_feature_found = true;
      best_feature_index = weighted_features.size();
      best_absolute_gradient = std::abs(gradient);
      new_feature = feature;
    }
  }
  if (new_feature_found) {
    weighted_features.push_back(std::make_pair(0.0, new_feature));
  }
  model_gradient = best_absolute_gradient;
  direction = best_feature_index;
}

// Sets step_size attribute of this model to a value defined by
// version 1 of DMaxEnt algorithm. Assumes that population and
// sample expectations of each feature are up to date. 
void DMaxEntModel::FindStepSize1() {
  double feature_weight = weighted_features[direction].first;
  Feature *feature = weighted_features[direction].second;
  double Phi_pt = lambda +
    (feature->GetUnnormalizedPopulationExpectation() / normalizer);
  double Phi_mt = -lambda +
    (feature->GetUnnormalizedPopulationExpectation() / normalizer);
  double Phi_p = lambda + feature->GetSampleExpectation();
  double Phi_m = -lambda + feature->GetSampleExpectation();
  double beta = 
    (Phi_pt * Phi_m * exp(-2 * feature_weight * lambda) - Phi_p * Phi_mt) /
    (Phi_pt * exp(-2 * feature_weight * lambda) - Phi_mt);
  double beta_k = 2 * model_parameter_alpha * feature->Complexity() +
    model_parameter_beta;
  if (std::abs(beta) < beta_k) {
    step_size = -feature_weight;
  } else if (beta > beta_k) {
    step_size = 0.5 * log(Phi_mt * (beta_k - Phi_p) /
			  (Phi_pt * (beta_k - Phi_m))) / lambda;
  } else {
    step_size = 0.5 * log(Phi_mt * (beta_k + Phi_p) /
			  (Phi_pt * (beta_k + Phi_m))) / lambda;
  }
}

// Sets step_size attribute of this model to a value defined by
// version 2 of DMaxEnt algorithm. Assumes that population and
// sample expectations of each feature are up to date. 
void DMaxEntModel::FindStepSize2() {
  double feature_weight = weighted_features[direction].first;
  Feature *feature = weighted_features[direction].second;
  double diff_expectations = -feature->GetSampleExpectation() +
    feature->GetUnnormalizedPopulationExpectation() / normalizer;
  double beta_k = 2 * model_parameter_alpha * feature->Complexity() +
    model_parameter_beta;
  double beta = feature_weight * lambda * lambda - diff_expectations; 
  if (std::abs(beta) <= beta_k) {
    step_size = -feature_weight;
  } else if (beta > beta_k) {
    step_size = - (beta_k + diff_expectations) / (lambda * lambda);
  } else {
    step_size = - (-beta_k + diff_expectations) / (lambda * lambda);
  }
}

// Updates this model, by computing new weights of each point in the space
// according to the last step of coordinate descent and sets appropriate
// value for the normalizer.
void DMaxEntModel::UpdateModel() {
  double weight;
  double new_normalizer = 0.0;
  weighted_features[direction].first += step_size;
  Feature *feature = weighted_features[direction].second;
  for (auto &point : *space) {
    weight = point.GetProbWeight() *
      exp(step_size * feature->FeatureMap(&point));
    new_normalizer += weight;
    point.SetProbWeight(weight);
  }
  normalizer = new_normalizer;
}

// Constructor for this model. Client needs to specify regularization
// parameters, maximum number of iterations of optimization procedure,
// version of step size formula, uniform bound on feature values,
// space (over which model is fit) and a set of features (used to fit
// this model).
DMaxEntModel::DMaxEntModel(double alpha, double beta,
			   double max_steps, int ver, double l,
			   bool stop_on_convergence, Space *X,
			   Sample S, std::vector<Feature*> *features,
			   std::vector<WLearner*> learners,
			   Sample test) {
  model_parameter_alpha = alpha;
  model_parameter_beta = beta;
  max_descent_steps = max_steps;
  version = ver;
  lambda = l;
  space = X;
  sample = S;
  normalizer = X->NumPoints();
  for (auto feature : *features) {
    weighted_features.push_back(std::make_pair(0.0, feature));
  }
  weak_learners = learners;
  stop_if_converged = stop_on_convergence;
  test_sample = test;
}

// Destructor for this model.
DMaxEntModel::~DMaxEntModel() {
  delete space;
  for (auto weight_feature_pair : weighted_features) {
    delete weight_feature_pair.second;
  }
  delete &weighted_features;
  for (auto learner : weak_learners) {
    delete learner;
  }
}

// Fits this model to the data using parameters which are specified
// during construction.
void DMaxEntModel::Fit() {
  for (int iter = 0; iter < max_descent_steps; iter++) {
    FindDescentDirection();
    if (version == 1) {
      FindStepSize1();
    } else {
      FindStepSize2();
    }
    UpdateModel();

    // log some statistics if needed
    VLOG(1) << "Completed iteration #" << iter + 1 <<
      " of coordinate descent: direction=" << direction <<
      " weight=" << step_size << " absolute gradient=" << model_gradient; 
    VLOG(2) << "Training Log loss: " << LogLoss(&sample);
    VLOG(3) << "Training AUC: " << AUC(&sample);

    // A little hack to also check for test error along the way
    VLOG(4) << "Test Log Loss: " << LogLoss(&test_sample);
    VLOG(4) << "Test AUC: " << AUC(&test_sample);

    if ((model_gradient < gTolerance) && stop_if_converged) {
      break;
    }
  }
}

// Returns the log loss of this model on the given sample.
double DMaxEntModel::LogLoss(Sample *sample) {
  double loss = 0.0;
  for (auto example : *sample) {
    loss += log(normalizer / example->GetProbWeight());
  }
  return loss;
}

// A functor to compare pointers to Points. The points are compared
// based on the probability weights and ties are broken based on
// positive examples supplied with positive examples ranked higher.
// TODO: this should be moved to Point class as a friend class.
class PointProbLessThanOperator : std::binary_function<Point*, Point*, bool> {
public:
  PointProbLessThanOperator(std::vector<bool> *positive) {
    positive_ = positive;
  }
  inline bool operator()(Point* point1, Point* point2) {
    return ((point1->GetProbWeight() < point2->GetProbWeight()) ||
	    ((point1->GetProbWeight() == point2->GetProbWeight()) &&
	     !(positive_->at(point1->GetId())) &&
	     (positive_->at(point2->GetId()))));
  }
private:
  std::vector<bool> *positive_;
};


// Returns AUC of this model on the given sample.
double DMaxEntModel::AUC(Sample *sample) {
  std::vector<bool> positive(space->NumPoints(), false);
  for (auto point : *sample) {
    positive[point->GetId()] = true;
  }
  std::vector<Point*> all_points;
  for (auto &point : *space) {
    all_points.push_back(&point);
  }

  std::sort(all_points.begin(), all_points.end(),
	    PointProbLessThanOperator(&positive));
  double n = 0.0;
  double r = 0.0;

  for (auto point : all_points) {
    if (positive[point->GetId()]) {
      r += n;
    } else {
      n += 1.0;
    }
  }
  // TODO: check that n > 0 and n < all_points.size()
  return r / (n * (all_points.size() - n));
}

// Returns the value stored in direction attribute. Typically this
// should be the last direction chosen by coordinate descent procedure.
// The value is undefined if Fit() method has not been called yet.
// This method is provided primarily for testing purposes.
int DMaxEntModel::GetDescentDirection() {
  return direction;
}

// Returns the value stored in step_size attribute. Typically this
// should be the last step size chosen by coordinate descent procedure.
// The value is undefined if Fit() method has not been called yet.
// This method is provided primarily for testing purposes.
double DMaxEntModel::GetStepSize() {
  return step_size;
}

// Returns the value stored in normalizer attribute. Typically this should
// be the value that normalizes point weights in space to be a probability
// distribution.
// This method is provided primarily for testing purposes.
double DMaxEntModel::GetNormalizer() {
  return normalizer;
}

// Returns a weight of the feature stored internally at specified index.
// If index is not specified correctly then returns -1. 
// This method is provided primarily for testing purposes.
double DMaxEntModel::GetWeight(int coordinate) {
  if (coordinate < 0 || coordinate >= weighted_features.size()) {
    return -1.0;
  }
  return weighted_features[coordinate].first;
}


// Returns iterator to the beginning of the feature vector
DMaxEntModel::FeatureIterator DMaxEntModel::FeatureBegin() {
  return weighted_features.begin();
}

// Returns iterator to the end of the feature vector
DMaxEntModel::FeatureIterator DMaxEntModel::FeatureEnd() {
  return weighted_features.end();
}
