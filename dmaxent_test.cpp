#include <cmath>
#include "gtest/gtest.h"
#include "constants.hpp"
#include "dmaxent.hpp"

// Test Feature for DMaxEntModel class.
class DMaxEntModelTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    point1 = new Point(1);
    point1->AddRawFeature(0.5);
    point1->AddRawFeature(0.4);
    point1->AddRawFeature(0.7);
    point1->AddRawFeature(0.7);
    point2 = new Point(2);
    point2->AddRawFeature(0.9);
    point2->AddRawFeature(0.1);
    point2->AddRawFeature(0.1);
    point2->AddRawFeature(0.1);
    space = new Space();
    space->AddPoint(*point1);
    space->AddPoint(*point2);
    space->Finalize();
    sample.push_back(point1);
    sample.push_back(point2);
    sample.push_back(point2);
    sample.push_back(point2);
    features.push_back(new RawFeature(0));
    features.push_back(new ProductFeature(1, 2));
    features.push_back(new ThresholdFeature(3, 0.5));
    features[0]->ComputeSampleExpectation(sample);
    features[1]->ComputeSampleExpectation(sample);
    features[2]->ComputeSampleExpectation(sample);
    features[0]->SetComplexity(0.0);
    features[1]->SetComplexity(0.0);
    features[2]->SetComplexity(0.0);
  }
  Point *point1;
  Point *point2;
  Space *space;
  Sample sample;
  std::vector<Feature*> features;
  DMaxEntModel *model;
  std::vector<WLearner*> learners;
  Sample test;
};

// Tests that fit method modifies the state correctly after 1 iteration
// where we update step size using version 1 formulation
TEST_F(DMaxEntModelTest, TestFitAfterOneIterationVer1) {
  model = new DMaxEntModel(0.0, 0.07, 1, 1, 1, true, space, sample, &features,
			   learners, test);
  model->Fit();
  EXPECT_EQ(2, model->GetDescentDirection());
  EXPECT_NEAR(-0.21765903562892275, model->GetStepSize(), gTolerance);
  EXPECT_NEAR(1.8043996665398437, model->GetNormalizer(), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(0), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(1), gTolerance);  
  EXPECT_NEAR(-0.21765903562892275, model->GetWeight(2), gTolerance);
  EXPECT_NEAR(0.8043996665398437, (space->GetPoint(0)).GetProbWeight(),
	      gTolerance);
  EXPECT_NEAR(1.0, (space->GetPoint(1)).GetProbWeight(), gTolerance);
}

// Tests that fit method modifies the state correctly after 2 iterations
// where we update step size using version 1 formulation
TEST_F(DMaxEntModelTest, TestFitAfterTwoIterationsVer1) {
  model = new DMaxEntModel(0.0, 0.07, 2, 1, 1, true, space, sample, &features,
			   learners, test);
  model->Fit();
  EXPECT_EQ(2, model->GetDescentDirection());
  EXPECT_NEAR(-0.1477979362200573, model->GetStepSize(), gTolerance);
  EXPECT_NEAR(1.6938794950220394, model->GetNormalizer(), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(0), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(1), gTolerance);  
  EXPECT_NEAR(-0.3654569718489801, model->GetWeight(2), gTolerance);
  EXPECT_NEAR(0.6938794950220394, (space->GetPoint(0)).GetProbWeight(),
	      gTolerance);
  EXPECT_NEAR(1.0, (space->GetPoint(1)).GetProbWeight(), gTolerance);
}

// Tests that fit method modifies the state correctly after 3 iterations
// where we update step size using version 1 formulation
TEST_F(DMaxEntModelTest, TestFitAfterThreeIterationsVer1) {
  model = new DMaxEntModel(0.0, 0.07, 3, 1, 1, true, space, sample, &features,
			   learners, test);
  model->Fit();
  EXPECT_EQ(2, model->GetDescentDirection());
  EXPECT_NEAR(-0.10353052379229892, model->GetStepSize(), gTolerance);
  EXPECT_NEAR(1.6256354062769477, model->GetNormalizer(), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(0), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(1), gTolerance);  
  EXPECT_NEAR(-0.468987495641279, model->GetWeight(2), gTolerance);
  EXPECT_NEAR(0.6256354062769477, (space->GetPoint(0)).GetProbWeight(),
	      gTolerance);
  EXPECT_NEAR(1.0, (space->GetPoint(1)).GetProbWeight(), gTolerance);
}


// Tests that fit method modifies the state correctly after 1 iteration
// where we update step size using version 2 formulation
TEST_F(DMaxEntModelTest, TestFitAfterOneIterationVer2) {
  model = new DMaxEntModel(0.0, 0.07, 1, 2, 1, true, space, sample, &features,
			   learners, test);
  model->Fit();
  EXPECT_EQ(2, model->GetDescentDirection());
  EXPECT_NEAR(-0.18, model->GetStepSize(), gTolerance);
  EXPECT_NEAR(1.835270211411272, model->GetNormalizer(), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(0), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(1), gTolerance);  
  EXPECT_NEAR(-0.18, model->GetWeight(2), gTolerance);
  EXPECT_NEAR(0.835270211411272, (space->GetPoint(0)).GetProbWeight(),
	      gTolerance);
  EXPECT_NEAR(1.0, (space->GetPoint(1)).GetProbWeight(), gTolerance);
}

// Tests that log loss method returns correct result.
TEST_F(DMaxEntModelTest, TestLogLoss) {
  model = new DMaxEntModel(0.0, 0.07, 3, 1, 1, true, space, sample, &features,
			   learners, test);
  model->Fit();
  Sample test_sample;
  test_sample.push_back(&(space->GetPoint(0)));
  test_sample.push_back(&(space->GetPoint(0)));
  test_sample.push_back(&(space->GetPoint(1)));
  test_sample.push_back(&(space->GetPoint(1)));
  test_sample.push_back(&(space->GetPoint(1)));
  EXPECT_NEAR(3.3674687842873072, model->LogLoss(&test_sample), gTolerance);
}

// Tests that AUC method returns correct result.
TEST_F(DMaxEntModelTest, TestAUC) {
  space = new Space();
  Point *new_point1 = new Point(0);
  new_point1->SetProbWeight(3);
  space->AddPoint(*new_point1);
  Point *new_point2 = new Point(1);
  new_point2->SetProbWeight(2);
  space->AddPoint(*new_point2);
  Point *new_point3 = new Point(2);
  new_point3->SetProbWeight(1.5);
  space->AddPoint(*new_point3);
  Point *new_point4 = new Point(3);
  new_point4->SetProbWeight(1.0);
  space->AddPoint(*new_point4);
  Point *new_point5 = new Point(4);
  new_point5->SetProbWeight(0.8);
  space->AddPoint(*new_point5);
  Point *new_point6 = new Point(5);
  new_point6->SetProbWeight(0.7);
  space->AddPoint(*new_point6);
  Point *new_point7 = new Point(6);
  new_point7->SetProbWeight(0.5);
  space->AddPoint(*new_point7);
  Point *new_point8 = new Point(7);
  new_point8->SetProbWeight(0.3);
  space->AddPoint(*new_point8);
  Point *new_point9 = new Point(8);
  new_point9->SetProbWeight(0.1);
  space->AddPoint(*new_point9);
  Point *new_point10 = new Point(9);
  new_point10->SetProbWeight(0.1);
  space->AddPoint(*new_point10);

  model = new DMaxEntModel(0.0, 0.07, 3, 1, 1, true, space, sample, &features,
			   learners, test);
  Sample test_sample;
  test_sample.push_back(new_point9);
  test_sample.push_back(new_point6);
  test_sample.push_back(new_point5);
  test_sample.push_back(new_point5);
  test_sample.push_back(new_point3);
  test_sample.push_back(new_point2);
  test_sample.push_back(new_point1);

  EXPECT_NEAR(0.79166666666, model->AUC(&test_sample), gTolerance);  
}

// Tests that fit method modifies the state correctly after 1 iteration
// where we update step size using version 1 formulation and
// set of weak learners is non empty and consists of one monomial
// and one tree learner
TEST_F(DMaxEntModelTest, TestFitWithWeakLearnersIterationVer1) {
  std::vector< std::map<double, double> > vtot;
  std::map<double, double> vtot1;
  vtot1[0.5] = 1.0;
  vtot1[0.9] = 1.0;
  vtot.push_back(vtot1);
  std::map<double, double> vtot2;
  vtot2[0.4] = 1.0;
  vtot2[0.1] = 1.0;
  vtot.push_back(vtot2);
  std::map<double, double> vtot3;
  vtot3[0.7] = 1.0;
  vtot3[0.1] = 0.5;
  vtot.push_back(vtot3);
  std::map<double, double> vtot4;
  vtot4[0.7] = 1.0;
  vtot4[0.1] = 1.0;
  vtot.push_back(vtot4);

  learners.push_back(new TreeLearner(4, 0.0, 0.07, vtot));
  learners.push_back(new MonomialLearner(4, 0.0, 0.07, 1.0));

  features.pop_back();
  model = new DMaxEntModel(0.0, 0.07, 1, 1, 1, true, space, sample, &features,
			   learners, test);
  model->Fit();
  EXPECT_EQ(2, model->GetDescentDirection());
  EXPECT_NEAR(-0.21765903562892275, model->GetStepSize(), gTolerance);
  EXPECT_NEAR(1.8043996665398437, model->GetNormalizer(), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(0), gTolerance);
  EXPECT_NEAR(0.0, model->GetWeight(1), gTolerance);  
  EXPECT_NEAR(-0.21765903562892275, model->GetWeight(2), gTolerance);
  EXPECT_NEAR(0.8043996665398437, (space->GetPoint(0)).GetProbWeight(),
  	      gTolerance);
  EXPECT_NEAR(1.0, (space->GetPoint(1)).GetProbWeight(), gTolerance);
}
