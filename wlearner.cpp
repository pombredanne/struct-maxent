#include <queue>
#include <cmath>
#include <map>
#include "constants.hpp"
#include "wlearner.hpp"
#include "space.hpp"
#include "feature.hpp"
#include "tree.hpp"

// Trains and returns a new Tree Feature based on given
// sample space and a sample. Returned tree feature is guaranteed
// to have sample and population expectations and complexity set
// to correct values. The gradient of the trained feature is also returned.
void TreeLearner::Train(Space &space, Sample &sample,
			Feature **feature, double *tree_gradient) {
  Node *root = new Node();
  for (auto &point : space) {
    root->AddPoint(&point);
  }
  for (auto point : sample) {
    root->AddSample(point);
  }
  root->SetValue(0);
  double old_diff = 0.0;
  int tree_size = 1;
  double old_gradient = 0.0;
  double normalizer = root->GetPopulationWeight();
  std::queue<Node*> q;
  q.push(root);
  while (!q.empty()) {
    Node *node = q.front();
    q.pop();
    double best_gradient = 0.0;
    double best_threshold = NAN;
    double best_feature_index = 0;
    double best_diff = 0.0;
    double best_left_val;
    for (int feature_index = 0; feature_index < num_features; feature_index++)
    {
      double threshold;
      double gradient;
      double diff;
      double left_val;
      BestThreshold(feature_index, node, old_diff, normalizer,
		    sample.size(), tree_size, &threshold, &gradient,
		    &left_val, &diff);
      if (std::abs(gradient) > std::abs(best_gradient) + gTolerance) {
	best_gradient = gradient;
	best_threshold = threshold;
	best_feature_index = feature_index;
	best_left_val = left_val;
	best_diff = diff;
      }
    }
    if (std::abs(best_gradient) > std::abs(old_gradient) +  gTolerance) {
      old_gradient = best_gradient;
      Node *left_child;
      Node *right_child;
      GrowTree(node, best_threshold, best_feature_index, best_left_val,
	       &left_child, &right_child);
      q.push(left_child);
      q.push(right_child);
      tree_size += 2;
      old_diff = best_diff;
    }
  }
  *tree_gradient = old_gradient;
  TreeFeature* tfeature = new TreeFeature(root);
  tfeature->ComputeTreeExpectations();
  tfeature->SetComplexity(TreeComplexity(tree_size, sample.size()));
  *feature = tfeature;
}

// Finds the best threshold (threshold with the largest absolute gradient)
// to split the given node based on the values of specified feature.
// The computation also requires the current size of the tree, sample size,
// the current expectation difference of the tree, normalizer for point
// weights. The results (threshold, gradient, new expectation difference
// and value of the left child) are returned via pointers.
void TreeLearner::BestThreshold(int feature_index, Node *node, double old_diff,
				double normalizer, int sample_size,
				int tree_size, double *threshold,
				double *grad, double *left_val, double *diff) {
  std::map<double, std::pair<double, int> > threshold_to_weights =
    BuildThresholdToWeightsMap(node, feature_index);
  double best_gradient = -1.0;
  double left_population_weight = 0.0;
  double right_population_weight = node->GetPopulationWeight();
  double left_sample_count = 0.0;
  double right_sample_count = double(node->GetSampleCount());
  for (const std::pair<double, std::pair<double, int>>& elem :
	 threshold_to_weights) {
    left_population_weight += elem.second.first;
    right_population_weight -= elem.second.first;
    left_sample_count += elem.second.second;
    right_sample_count -= elem.second.second;
    double left_diff = (left_population_weight / normalizer
			- left_sample_count / sample_size);
    double right_diff = (right_population_weight / normalizer
			 - right_sample_count / sample_size);
    double led = old_diff + (1 - 2 * node->GetValue()) * left_diff;
    double red = old_diff + (1 - 2 * node->GetValue()) * right_diff;
    double new_left_gradient = Gradient(tree_size + 2, sample_size, led);
    double new_right_gradient = Gradient(tree_size + 2, sample_size, red);
    double new_gradient = (std::abs(new_left_gradient) >
			   std::abs(new_right_gradient) + gTolerance ?
			   new_left_gradient : new_right_gradient);
    if (std::abs(new_gradient) > best_gradient +
	gTolerance) {
      best_gradient = std::abs(new_gradient);
      *grad = new_gradient;
      *threshold = elem.first;
      *left_val = (std::abs(new_left_gradient) > std::abs(new_right_gradient)
		   + gTolerance ? 1.0 - node->GetValue() : node->GetValue());
      *diff = (std::abs(new_left_gradient) > std::abs(new_right_gradient)
	       + gTolerance ? led : red);
    }
  }
}

// Returns the value of the gradient of the structural maxent
// objective for the tree with specified tree size and expectation difference
// trained on a sample with a given sample size.
double TreeLearner::Gradient(int tree_size, int sample_size,
			     double expectation_difference) {
  double complexity = model_parameter_beta +
    model_parameter_alpha * TreeComplexity(tree_size, sample_size);
  if (std::abs(expectation_difference) < complexity) {
    return 0;
  } else {
    int sgn = (expectation_difference > 0 ? 1 : -1);
    return expectation_difference - sgn * complexity;
  }
}

// Returns the complexity of the tree with specified sample size
// and tree size.
double TreeLearner::TreeComplexity(int tree_size, int sample_size) {
  return sqrt((4 * tree_size + 2) *
	      (log(double(num_features) + 2.0) / log(2.0)) *
	      (log(double(sample_size) + 1.0)) /
	      (double(sample_size)));
}


// Grows a tree at a given node. This node recieves left and right child
// and pointers to these nodes are returned via corresponding variables.
// The value of the left child is given and the value of the right one
// one minus that value.
// Node also updates its threshold and raw feature to the specified ones.
// All samples and points stored in this node with raw feature less than
// the threshold are moved to the left child and the rest are moved to
// the left child.
void TreeLearner::GrowTree(Node *node, double threshold,
			   int feature_index, int left_val,
			   Node **left_child, Node **right_child) {
  *left_child = new Node();
  *right_child = new Node();
  node->SetThreshold(threshold);
  node->SetFeature(feature_index);
  node->SetLeftChild(*left_child);
  node->SetRightChild(*right_child);
  (*left_child)->SetValue(left_val);
  (*right_child)->SetValue(1-left_val);
  for (std::vector<Point*>::iterator it = node->PointsBegin();
       it != node->PointsEnd(); it++) {
    Point *point = *it;
    if (point->GetRawFeature(feature_index) < threshold) {
      (*left_child)->AddPoint(point);
    } else {
      (*right_child)->AddPoint(point);
    }
  }
  for (std::vector<Point*>::iterator it = node->SamplesBegin();
       it != node->SamplesEnd(); it++) {
    Point *point = *it;
    if (point->GetRawFeature(feature_index) < threshold) {
      (*left_child)->AddSample(point);
    } else {
      (*right_child)->AddSample(point);
    }
  }
  node->ClearPoints();
  node->ClearSamples();
}

// Returns a map from thresholds to weight-count pairs. The weight
// is the total population weight of points in this node with a feature
// at a given index between this threshold the next smallest one. The count
// is the total number of sample points in this node with a feature at
// a given index between this threshold the next smallest one.
std::map<double, std::pair<double, int> >
TreeLearner::BuildThresholdToWeightsMap(Node *node, int index) {
  std::map<double, std::pair<double, int> > threshold_to_weights;
  for (std::vector<Point*>::iterator it = node->PointsBegin();
       it != node->PointsEnd(); it++) {
    Point *point = *it;
    threshold_to_weights[value_to_thresholds
			 [index][point->GetRawFeature(index)]].first +=
      point->GetProbWeight();
  }
  for (std::vector<Point*>::iterator it = node->SamplesBegin();
       it != node->SamplesEnd(); it++) {
    Point *point = *it;
    threshold_to_weights[value_to_thresholds
			 [index][point->GetRawFeature(index)]].second += 1;
  }
  return threshold_to_weights;
} 

// Constructs Tree Learner with specified parameters
TreeLearner::TreeLearner(int n_features, double alpha, double beta,
			 std::vector< std::map<double, double> > vtot) {
  num_features = n_features;
  model_parameter_alpha = alpha;
  model_parameter_beta = beta;
  value_to_thresholds = vtot;
}

// Trains and returns a new Monomial Feature based on given
// sample space and sample. Returned feature is guaranteed
// to have sample and population expectations and complexity set
// to correct values. The gradient of the trained feature is also returned.
void MonomialLearner::Train(Space &space, Sample &sample,
			    Feature **feature, double *monomial_gradient) {
  std::vector<int> monomial(num_features);
  std::vector<double> point_values;
  double normalizer = 0.0;
  double best_gradient = 0.0;
  double monomial_population_expectation;
  double monomial_sample_expectation;
  int power = 0;
  for (auto &point : space) {
    point_values.push_back(point.GetProbWeight());
    normalizer += point.GetProbWeight();
  }
  std::vector<double> sample_values(sample.size(), 1.0);

  bool stop = false;
  while (!stop) {
    double candidate_gradient;
    int candidate_feature;
    double candidate_population_expectation;
    double candidate_sample_expectation;
    BestFeature(point_values, sample_values, space, sample, normalizer, power,
		&candidate_gradient, &candidate_feature,
		&candidate_population_expectation,
		&candidate_sample_expectation);
    if (std::abs(candidate_gradient) > std::abs(best_gradient) + gTolerance) {
      // update best gradient and monomial
      best_gradient = candidate_gradient;
      monomial_population_expectation = candidate_population_expectation;
      monomial_sample_expectation = candidate_sample_expectation;
      monomial[candidate_feature] += 1;
      power += 1;
      int index = 0;
      for (auto &point : space) {
	point_values[index] *= point.GetRawFeature(candidate_feature);
	index++;
      }
      index = 0;
      for (auto point : sample) {
	sample_values[index] *= point->GetRawFeature(candidate_feature);
	index++;
      }
    } else { // no improvment - stop and return monomial
      stop = true;
    }
    MonomialFeature *mfeature = new MonomialFeature(monomial);
    mfeature->SetComplexity(MonomialComplexity(power, sample.size()));
    mfeature->MonomialExpectations(monomial_population_expectation,
				  monomial_sample_expectation);
    *monomial_gradient = best_gradient;
    *feature = mfeature;
  }
}

// Returns the value of the gradient of the structural maxent
// objective for the monomial with specified power and expectation difference
// trained on a sample with a given sample size.
double MonomialLearner::Gradient(int power, int sample_size,
				 double difference) {
  double complexity = model_parameter_beta +
    model_parameter_alpha * MonomialComplexity(power, sample_size);
  if (std::abs(difference) < complexity) {
    return 0;
  } else {
    int sgn = (difference > 0 ? 1 : -1);
    return difference - sgn * complexity;
  }
}

// Returns the complexity of the monomial with specified sample size
// and power.
double MonomialLearner::MonomialComplexity(int power, int sample_size) {
  return sqrt(2 * feature_bound * double(power) *
	      log(double(num_features)) / double(sample_size));
}

// Finds the best raw feature (in the sense of the largest absolute gradient)
// to add to the monomial based on the values of the monomial
// at each space and sample point (specified in point_values and
// sample_values). Note that point values need to be weighted by
// by the corresponding (unnormalized) point weights.
// The computation also requires the current power of monomial
// and normalizer for point weights.
// The results (best gradient and feature) are returned via pointers.
void MonomialLearner::BestFeature(const std::vector<double> &point_values,
				  const std::vector<double> &sample_values,
				  Space &space, Sample &sample,
				  double normalizer, int power,
				  double *candidate_gradient,
				  int *candidate_feature,
				  double *candidate_population_expectation,
				  double *candidate_sample_expectation) {
    *candidate_gradient = 0.0;
    *candidate_feature = 0;
    *candidate_population_expectation = 0.0;
    *candidate_sample_expectation = 0.0;
    for (int feature = 0; feature < num_features; feature++) {
      double population_expectation = 0.0;
      int index = 0;
      for (auto &point : space) {
	population_expectation += point_values[index] *
	  point.GetRawFeature(feature);
	index++;
      }
      population_expectation /= normalizer;
      index = 0;
      double sample_expectation = 0.0;
      for (auto point : sample) {
	sample_expectation += sample_values[index] *
	  point->GetRawFeature(feature);
	index++;
      }
      sample_expectation /= sample.size();
      double diff = population_expectation - sample_expectation;
      double gradient = Gradient(power + 1, sample.size(), diff);
      if (std::abs(gradient) > std::abs(*candidate_gradient) + gTolerance) {
	*candidate_gradient = gradient;
	*candidate_feature = feature;
	*candidate_population_expectation = population_expectation;
	*candidate_sample_expectation = sample_expectation;
      }	
    }
}

// Constructs Monomial Learner with specified parameters.
MonomialLearner::MonomialLearner(int n_features, double alpha, double beta,
				 double bound) {
  num_features = n_features;
  model_parameter_alpha = alpha;
  model_parameter_beta = beta;
  feature_bound = bound;
}
