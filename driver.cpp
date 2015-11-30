
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "dmaxent.hpp"
#include "space.hpp"
#include "feature.hpp"
#include "constants.hpp"
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <cmath>
#include <algorithm>

DEFINE_double(model_parameter_alpha, 0.0, "Regularization parameter alpha.");
DEFINE_double(model_parameter_beta, 1.0, "Regularization parameter beta.");
DEFINE_int32(num_iterations, 1, "Number of iterations for optimization.");
DEFINE_int32(dmaxent_version, 1,
	     "Version of DMaxEnt algorithm used for optimization.");
DEFINE_double(feature_bound, 1.0,
	      "Uniform bound bound on features used.");
DEFINE_string(data_path, "", "Path to a file with the data set.");
DEFINE_int32(seed, 1, "Seed for random number generator.");
DEFINE_int32(train_size, 1, "Size of the training set.");
DEFINE_int32(num_bins, 10, "Number of bins used for threshold features.");
DEFINE_bool(raw, false, "If true raw features are used.");
DEFINE_bool(prod, false, "If true product features are used.");
DEFINE_bool(th, false, "If true threshold features are used.");
DEFINE_bool(mon, false, "If true monomial features are used.");
DEFINE_bool(tr, false, "If true tree features are used.");
DEFINE_bool(stop_if_converged, true, "If true coordinate descent will "
	    "terminate once gradient is sufficiently small.");

// Aborts the application if one of the flags has illegal value.
void ValidateFlags() {
  CHECK_GE(FLAGS_model_parameter_alpha, 0);
  CHECK_GE(FLAGS_model_parameter_beta, 0);
  CHECK_GE(FLAGS_num_iterations, 1);
  CHECK_GE(FLAGS_train_size, 1);
  CHECK_GE(FLAGS_num_bins, 2);
  CHECK(FLAGS_dmaxent_version == 1 || FLAGS_dmaxent_version == 2);
  CHECK_GE(FLAGS_feature_bound, 0);
  CHECK(!FLAGS_data_path.empty());
  CHECK(FLAGS_raw || FLAGS_prod || FLAGS_th || FLAGS_mon || FLAGS_tr);
}

// Splits a given string using specified delimeter character and
// stores results in a the given vector of strings.
// Consecutive delimeter characters are treated as a single delimeter.
// Example: split("std::vector", ':', elems) stores "std" and "vecctor"
// in elems.
void split(const std::string &s, const char delim,
	   std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      elems.push_back(item);
    }
  }
}

// A functor to compare pointers to Points. The points are compared
// based on the raw feature values at an index that defines this
// functor.
class PointLessThanOperator : std::binary_function<Point*, Point*, bool> {
public:
  PointLessThanOperator(int index) {
    raw_feature_index = index;
  }
  bool operator()(Point* point1, Point* point2) {
    return (point1->GetRawFeature(raw_feature_index) <
	    point2->GetRawFeature(raw_feature_index));
  }
private:
  int raw_feature_index;
};

// Reads in data from a file specified by the given file name.
// Each line in the file is assumed to contain a data on a particular
// point in space in the following format:
//   feature_value_1 .... feature_value_k num_of_observations_at_this_point
// Stores the points in the provided space and adds appropriate
// observations to the sample. Sample is split between training and
// testing according to specified value.
// Builds raw, product and threshold features, computes their expectations
// and stores them in feature vector.
void ReadData(std::string filename, int train_size,
	      Space *space, std::vector<Feature*> *features,
	      std::vector<WLearner*> *weak_learners,
	      Sample *train_sample, Sample *test_sample) {
  std::ifstream file(filename);
  CHECK(file.is_open());
  std::string line;
  std::vector<std::string> elems;
  Point *point;
  std::vector<int> counts;
  int point_count = 0;
  bool missing;

  // Read in data from a file
  while (!std::getline(file, line).eof()) {
    elems.clear();
    split(line, ' ', elems);
    point = new Point(point_count);
    missing = false;
    for (unsigned index = 0; index < elems.size() - 1; index++) {
      if (elems[index] == ".") {
	missing = true;
	continue;
      }
      point->AddRawFeature(atof(elems[index].c_str()));
    }
    if (missing) {
      continue;
    }
    counts.push_back(atoi((elems.back()).c_str()));
    space->AddPoint(*point);
    point_count++;
  }
  space->Finalize();
  
  // Partition sample points randomly into training and testing
  Sample all_sample;
  for (unsigned index = 0; index < counts.size(); index++) {
    for (int unused = 0; unused < counts[index]; unused++) {
      all_sample.push_back(&(space->GetPoint(index)));
    }
  }

  std::random_shuffle(all_sample.begin(), all_sample.end());
  for (unsigned index = 0; index < all_sample.size(); index++) {
    if (index < train_size) {
      train_sample->push_back(all_sample[index]);
    } else {
      test_sample->push_back(all_sample[index]);
    }
  }

  int num_raw_features = 0;
  if (point_count > 0) {
    num_raw_features = (space->GetPoint(0)).NumRawFeatures();
  }

  // Add raw features
  if (FLAGS_raw) {
    RawFeature *raw_feature;
    for (unsigned index = 0; index < num_raw_features; index++) {
      raw_feature = new RawFeature(index);
      raw_feature->ComputeSampleExpectation(*train_sample);
      features->push_back(raw_feature);
    }
    if (num_raw_features > 0) {
      raw_feature->SetComplexity(std::sqrt(2 * std::log(num_raw_features) /
					   train_size));
    }
  }

  // Add product features
  if (FLAGS_prod) {
    ProductFeature *prod_feature;
    for (unsigned index_i = 0; index_i < num_raw_features; index_i++) {
      for (unsigned index_j = 0; index_j < num_raw_features; index_j++) {
	prod_feature = new ProductFeature(index_i, index_j);
	prod_feature->ComputeSampleExpectation(*train_sample);
	features->push_back(prod_feature);
      }
    }
    if (num_raw_features > 0) {
      prod_feature->SetComplexity(std::sqrt(4 * std::log(num_raw_features) /
					    train_size));
    }
  }

  // Add monomial weak learner
  if (FLAGS_mon) {
    weak_learners->push_back(new MonomialLearner(num_raw_features,
						 FLAGS_model_parameter_alpha,
						 FLAGS_model_parameter_beta,
						 FLAGS_feature_bound));
  }

  // Add threshold features or tree weak learner
  std::vector<Point*> all_points;
  for (unsigned index = 0; index < point_count; index++) {
    all_points.push_back(&(space->GetPoint(index)));
  }

  if (FLAGS_tr || FLAGS_th) {

    // Thresholds are chosen so that resulting bins have
    // (approximately) same number of points.
    // The difficulty is that feature values need not be unique
    int bin_size = point_count / FLAGS_num_bins;
    int bin_count;
    double previous_value, value, current_value;
    Feature* threshold_feature;
    int threshold_feature_count = 0;
    std::vector<std::vector<double>> thresholds;
    for (unsigned index = 0; index < num_raw_features; index++) {
      std::sort(all_points.begin(), all_points.end(),
		PointLessThanOperator(index));
      bin_count = 0;
      current_value = all_points.at(0)->GetRawFeature(index);
      previous_value = current_value;
      std::vector<double> thresholds_for_this_feature;
      VLOG(1) << "Thresholds for feature #" << index << ":";
      for (auto point : all_points) {
	value = point->GetRawFeature(index);
	// Every time we have observed bin_size elements we put
	// a threshold. The only exception is when we have not seen
	// new values since the last threshold.
	if ((bin_count > bin_size) && (value != previous_value)) {
	  if (FLAGS_th) {
	    threshold_feature =
	      new ThresholdFeature(index, 0.5 * (value + previous_value));
	    threshold_feature->ComputeSampleExpectation(*train_sample);
	    features->push_back(threshold_feature);
	    threshold_feature_count++;
	  }
	  if (FLAGS_tr) {
	    thresholds_for_this_feature.push_back(0.5 *
						  (value + previous_value));
	  }
	  VLOG(1) << 0.5 * (value + previous_value);
	  current_value = value;
	  previous_value = current_value;
	  bin_count = 0; // will be reset to one at the end of iteration
	}
	if (value != current_value) {
	  previous_value = current_value;
	  current_value = value;
	}
	bin_count++;
      }
      thresholds_for_this_feature.push_back(FLAGS_feature_bound + 1);
      thresholds.push_back(thresholds_for_this_feature);
      thresholds_for_this_feature.clear();
    }
    if (threshold_feature_count > 0) {
      threshold_feature->
	SetComplexity(std::sqrt(2 * std::log(threshold_feature_count) /
				train_size));
    }
    if (FLAGS_tr) {
      // TODO: this inefficient - need to do it in one loop
      std::vector< std::map<double, double> > values_to_thresholds;
      std::map<double, double> vtot;
      for (unsigned feature = 0; feature < num_raw_features; feature++) {
	std::sort(all_points.begin(), all_points.end(),
		  PointLessThanOperator(feature));
	int next_threshold = 0;
	//printf("feature=%d: %f ", feature, thresholds[feature][next_threshold]);
	for (auto point : all_points) {
	  if (point->GetRawFeature(feature) >
	      thresholds[feature][next_threshold]) {
	    next_threshold++;
	    //printf("%f ", thresholds[feature][next_threshold]);
	  }
	  vtot[point->GetRawFeature(feature)] =
	    thresholds[feature][next_threshold];
	}
	//printf("\n");
	values_to_thresholds.push_back(vtot);
	vtot.clear();
      }
      weak_learners->push_back(new TreeLearner(num_raw_features,
					       FLAGS_model_parameter_alpha,
					       FLAGS_model_parameter_beta,
					       values_to_thresholds));
    }
  }

  // Log some of the statistics
  VLOG(1) << "Number of (active) points: " << all_points.size();
  VLOG(1) << "Number of raw features: " << num_raw_features;
  VLOG(1) << "Number of all features: " << features->size();
  VLOG(1) << "Number of sample points: " << all_sample.size();

}

// Reads in data specified by the user, fits dmaxent model
// and evaluates the fit.
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  ValidateFlags();

  srand(FLAGS_seed);

  Space *space = new Space();
  std::vector<Feature*> features;
  std::vector<WLearner*> weak_learners;
  Sample train_sample;
  Sample test_sample;
  ReadData(FLAGS_data_path, FLAGS_train_size, space, &features, &weak_learners,
	   &train_sample, &test_sample);
  DMaxEntModel *model = new DMaxEntModel(FLAGS_model_parameter_alpha,
					 FLAGS_model_parameter_beta,
  					 FLAGS_num_iterations,
  					 FLAGS_dmaxent_version,
  					 FLAGS_feature_bound,
					 FLAGS_stop_if_converged,
  					 space,
					 train_sample,
  					 &features,
					 weak_learners,
					 test_sample);
  model->Fit();

  double model_log_loss = model->LogLoss(&test_sample);
  double model_AUC = model->AUC(&test_sample);


  VLOG(1) << "Model test log loss: " << model_log_loss;
  VLOG(1) << "Model test AUC: " << model_AUC;
  printf("Model log loss: %f\n", model_log_loss);
  printf("Model AUC: %f\n", model_AUC);

  if (VLOG_IS_ON(2)) {
    // do some more logging here
    VLOG(2) << "Features included in the model ... ";
    int index = 0;
    int count = 0;
    int n_tr = 0;
    int n_mon = 0;
    int n_raw = 0;
    int n_prod = 0;
    int n_th = 0;
    int tr_size = 0;
    int mon_power = 0;
    double tr_complexity = 0.0;
    double mon_complexity = 0.0;
    double raw_complexity = 0.0;
    double prod_complexity = 0.0;
    double th_complexity = 0.0;

    for (DMaxEntModel::FeatureIterator it = model->FeatureBegin() ;
	 it != model->FeatureEnd(); it++) {
      std::pair<double, Feature*> weighted_feature = *it;
      if (std::abs(weighted_feature.first) > gTolerance) {
	VLOG(2) << "Feature #" << index << " is included with weight="
		<< weighted_feature.first;
	count++;
	TreeFeature *tfeature =
	  dynamic_cast<TreeFeature*>(weighted_feature.second);
	if (tfeature != NULL) {
	  n_tr += 1;
	  tr_size += tfeature->TreeSize();
	  tr_complexity += tfeature->Complexity();
	  VLOG(2) << "This is a tree feature with size="
		  << tfeature->TreeSize() << " and complexity="
		  << tfeature->Complexity();
	}
	MonomialFeature *mfeature = 
	  dynamic_cast<MonomialFeature*>(weighted_feature.second);
	if (mfeature != NULL) {
	  n_mon += 1;
	  mon_power += mfeature->GetPower();
	  mon_complexity += mfeature->Complexity();
	  VLOG(2) << "This is a monomial feature with size="
		  << mfeature->GetPower() << " and complexity="
		  << mfeature->Complexity();
	}
	RawFeature *rfeature = 
	  dynamic_cast<RawFeature*>(weighted_feature.second);
	if (rfeature != NULL) {
	  n_raw += 1;
	  raw_complexity += rfeature->Complexity();
	  VLOG(2) << "This is a raw feature with complexity="
		  << rfeature->Complexity();
	}
	ProductFeature *prfeature = 
	  dynamic_cast<ProductFeature*>(weighted_feature.second);
	if (prfeature != NULL) {
	  n_prod += 1;
	  prod_complexity += prfeature->Complexity();
	  VLOG(2) << "This is a product feature with complexity="
		  << prfeature->Complexity();
	}
	ThresholdFeature *thfeature = 
	  dynamic_cast<ThresholdFeature*>(weighted_feature.second);
	if (thfeature != NULL) {
	  n_th += 1;
	  th_complexity += rfeature->Complexity();
	  VLOG(2) << "This is a threshold feature with complexity="
		  << thfeature->Complexity();
	}
	
      }
      index++;
    }
    VLOG(2) << "Total number of features included: " << count;
    VLOG(2) << "Total number of tree features included: " << n_tr;
    VLOG(2) << "Total number of monomial features included: " << n_mon;
    VLOG(2) << "Total number of raw features included: " << n_raw;
    VLOG(2) << "Total number of product features included: " << n_prod;
    VLOG(2) << "Total number of threshold features inclued: " << n_th;
    VLOG(2) << "Overal complexity of the model: " <<
      tr_complexity + mon_complexity + prod_complexity + th_complexity +
      raw_complexity;
    VLOG(2) << "Overal complexity of tree features: " << tr_complexity;
    VLOG(2) << "Overal complexity of monomial features: " << mon_complexity;
    VLOG(2) << "Overal complexity of raw features: " <<  raw_complexity;
    VLOG(2) << "Overal complexity of product features: " << prod_complexity;
    VLOG(2) << "Overal complexity of threshold features: " << th_complexity;


    if (n_tr > 0) {
      VLOG(2) << "Average size of tree features: "
	      << double(tr_size) / n_tr;
    }
    if (n_mon > 0) {
      VLOG(2) << "Average degree of monomial features: "
	      << double(mon_power) / n_mon;
    }

  }

  return 0;
}
