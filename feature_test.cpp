#include <cmath>
#include "gtest/gtest.h"
#include "space.hpp"
#include "constants.hpp"
#include "feature.hpp"

// Tests feature map for raw features
TEST(FeatureTest, TestRawFeatureMap) {
  RawFeature *feature1 = new RawFeature(0);
  RawFeature *feature2 = new RawFeature(1);
  RawFeature *feature3 = new RawFeature(2);
  RawFeature *feature4 = new RawFeature(3);
  Point *point = new Point(1);
  point->AddRawFeature(-10);
  point->AddRawFeature(0.5);
  point->AddRawFeature(123);
  EXPECT_NEAR(-10.0, feature1->FeatureMap(point), gTolerance);
  EXPECT_NEAR(0.5, feature2->FeatureMap(point), gTolerance);
  EXPECT_NEAR(123.0, feature3->FeatureMap(point), gTolerance);
  EXPECT_TRUE(isnan(feature4->FeatureMap(point)));
}

// Tests feature map for product features
TEST(FeatureTest, TestProductFeatureMap) {
  ProductFeature *feature1 = new ProductFeature(0, 0);
  ProductFeature *feature2 = new ProductFeature(1, 2);
  ProductFeature *feature3 = new ProductFeature(0, 2);
  ProductFeature *feature4 = new ProductFeature(3, 1);
  Point *point = new Point(1);
  point->AddRawFeature(-10);
  point->AddRawFeature(0.5);
  point->AddRawFeature(123);
  EXPECT_NEAR(100.0, feature1->FeatureMap(point), gTolerance);
  EXPECT_NEAR(61.5, feature2->FeatureMap(point), gTolerance);
  EXPECT_NEAR(-1230.0, feature3->FeatureMap(point), gTolerance);
  EXPECT_TRUE(isnan(feature4->FeatureMap(point)));
}

// Tests feature map for threshold features
TEST(FeatureTest, TestThresholdFeatureMap) {
  ThresholdFeature *feature1 = new ThresholdFeature(0, 0);
  ThresholdFeature *feature2 = new ThresholdFeature(1, 0);
  ThresholdFeature *feature3 = new ThresholdFeature(2, 200);
  ThresholdFeature *feature4 = new ThresholdFeature(3, 1);
  Point *point = new Point(1);
  point->AddRawFeature(-10);
  point->AddRawFeature(0.5);
  point->AddRawFeature(123);
  EXPECT_NEAR(0.0, feature1->FeatureMap(point), gTolerance);
  EXPECT_NEAR(1.0, feature2->FeatureMap(point), gTolerance);
  EXPECT_NEAR(0.0, feature3->FeatureMap(point), gTolerance);
  EXPECT_NEAR(0.0, feature4->FeatureMap(point), gTolerance);
}

// Tests feature map for tree features
TEST(FeatureTest, TestTreeFeatureMap) {
  Node *root = new Node();
  root->SetLeftChild(new Node());
  root->SetRightChild(new Node());
  root->GetLeftChild()->SetLeftChild(new Node());
  root->GetLeftChild()->SetRightChild(new Node());
  root->GetRightChild()->SetLeftChild(new Node());
  root->GetRightChild()->SetRightChild(new Node()); 
  root->SetThreshold(0.0);
  root->SetFeature(1);
  root->GetLeftChild()->SetThreshold(1.0);
  root->GetLeftChild()->SetFeature(2);
  root->GetLeftChild()->GetLeftChild()->SetValue(1);
  root->GetLeftChild()->GetRightChild()->SetValue(0);
  root->GetRightChild()->SetThreshold(-2.0);
  root->GetRightChild()->SetFeature(3);
  root->GetRightChild()->GetLeftChild()->SetValue(0);
  root->GetRightChild()->GetRightChild()->SetValue(1);
  TreeFeature *feature = new TreeFeature(root);
  Point *point1 = new Point(1);
  point1->AddRawFeature(1);
  point1->AddRawFeature(-1.0);
  point1->AddRawFeature(0.0);
  point1->AddRawFeature(123);
  Point *point2 = new Point(2);
  point2->AddRawFeature(1);
  point2->AddRawFeature(-0.5);
  point2->AddRawFeature(123);
  point2->AddRawFeature(3);
  Point *point3 = new Point(2);
  point3->AddRawFeature(-1);
  point3->AddRawFeature(0.0);
  point3->AddRawFeature(40);
  point3->AddRawFeature(-3);
  Point *point4 = new Point(2);
  point4->AddRawFeature(-1);
  point4->AddRawFeature(0.1);
  point4->AddRawFeature(-40);
  point4->AddRawFeature(1);
  EXPECT_NEAR(1.0, feature->FeatureMap(point1), gTolerance);
  EXPECT_NEAR(0.0, feature->FeatureMap(point2), gTolerance);
  EXPECT_NEAR(0.0, feature->FeatureMap(point3), gTolerance);
  EXPECT_NEAR(1.0, feature->FeatureMap(point4), gTolerance);
}

// Tests feature map for monomial feature
TEST(FeatureTest, TestMonomialFeatureMap) {
  int mon1[3] = {0, 2, 3};
  int mon2[3] = {3, 3, 0};
  int mon3[3] = {2, 1, 1};
  int mon4[3] = {1, 0, 2};
  std::vector<int> monomial1(mon1, mon1 + 3);
  std::vector<int> monomial2(mon2, mon2 + 3);
  std::vector<int> monomial3(mon3, mon3 + 3);
  std::vector<int> monomial4(mon4, mon4 + 3);

  MonomialFeature *feature1 = new MonomialFeature(monomial1);
  MonomialFeature *feature2 = new MonomialFeature(monomial2);
  MonomialFeature *feature3 = new MonomialFeature(monomial3);
  MonomialFeature *feature4 = new MonomialFeature(monomial4);
  Point *point = new Point(1);
  point->AddRawFeature(-1);
  point->AddRawFeature(0.5);
  point->AddRawFeature(3);
  EXPECT_NEAR(6.75, feature1->FeatureMap(point), gTolerance);
  EXPECT_NEAR(-0.125, feature2->FeatureMap(point), gTolerance);
  EXPECT_NEAR(1.5, feature3->FeatureMap(point), gTolerance);
  EXPECT_NEAR(-9.0, feature4->FeatureMap(point), gTolerance);
}


// Tests computing and getting sample expectation of a raw feature
TEST(FeatureTest, TestComputeGetSampleExpectationOfRawFeature) {
  RawFeature *feature = new RawFeature(1);
  EXPECT_TRUE(isnan(feature->GetSampleExpectation()));
  Point *point = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Sample sample;
  point->AddRawFeature(-10);
  point->AddRawFeature(0.5);
  point->AddRawFeature(123);
  point2->AddRawFeature(-10);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(123);
  point3->AddRawFeature(-10);
  point3->AddRawFeature(1.5);
  point3->AddRawFeature(123);
  sample.push_back(point);
  sample.push_back(point2);
  sample.push_back(point3);
  sample.push_back(point2);
  feature->ComputeSampleExpectation(sample);
  EXPECT_NEAR(1.0, feature->GetSampleExpectation(), gTolerance);
}

// Tests computing and getting sample expectation of a product feature
TEST(FeatureTest, TestComputeGetSampleExpectationOfProductFeature) {
  ProductFeature *feature = new ProductFeature(1, 1);
  EXPECT_TRUE(isnan(feature->GetSampleExpectation()));
  Point *point = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Sample sample;
  point->AddRawFeature(-10);
  point->AddRawFeature(3.0);
  point->AddRawFeature(123);
  point2->AddRawFeature(-10);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(123);
  point3->AddRawFeature(-10);
  point3->AddRawFeature(3.0);
  point3->AddRawFeature(123);
  sample.push_back(point);
  sample.push_back(point2);
  sample.push_back(point3);
  sample.push_back(point2);
  feature->ComputeSampleExpectation(sample);
  EXPECT_NEAR(5.0, feature->GetSampleExpectation(), gTolerance);
}

// Tests computing and getting sample expectation of a threshold feature
TEST(FeatureTest, TestComputeGetSampleExpectationOfThresholdFeature) {
  ThresholdFeature *feature = new ThresholdFeature(1, 1.1);
  EXPECT_TRUE(isnan(feature->GetSampleExpectation()));
  Point *point = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Sample sample;
  point->AddRawFeature(-10);
  point->AddRawFeature(3.0);
  point->AddRawFeature(123);
  point2->AddRawFeature(-10);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(123);
  point3->AddRawFeature(-10);
  point3->AddRawFeature(3.0);
  point3->AddRawFeature(123);
  sample.push_back(point);
  sample.push_back(point2);
  sample.push_back(point3);
  sample.push_back(point2);
  feature->ComputeSampleExpectation(sample);
  EXPECT_NEAR(0.5, feature->GetSampleExpectation(), gTolerance);
}

// Tests computing and getting sample expectation of a tree feature
TEST(FeatureTest, TestComputeGetSampleExpectationOfTreeFeature) {
  Node *root = new Node();
  root->SetLeftChild(new Node());
  root->SetRightChild(new Node());
  root->GetLeftChild()->SetLeftChild(new Node());
  root->GetLeftChild()->SetRightChild(new Node());
  root->GetRightChild()->SetLeftChild(new Node());
  root->GetRightChild()->SetRightChild(new Node()); 
  root->SetThreshold(0.0);
  root->SetFeature(1);
  root->GetLeftChild()->SetThreshold(1.0);
  root->GetLeftChild()->SetFeature(2);
  root->GetLeftChild()->GetLeftChild()->SetValue(1);
  root->GetLeftChild()->GetRightChild()->SetValue(0);
  root->GetRightChild()->SetThreshold(-2.0);
  root->GetRightChild()->SetFeature(3);
  root->GetRightChild()->GetLeftChild()->SetValue(0);
  root->GetRightChild()->GetRightChild()->SetValue(1);
  TreeFeature *feature = new TreeFeature(root);
  EXPECT_TRUE(isnan(feature->GetSampleExpectation()));
  Point *point1 = new Point(1);
  point1->AddRawFeature(1);
  point1->AddRawFeature(-1.0);
  point1->AddRawFeature(0.0);
  point1->AddRawFeature(123);
  Point *point2 = new Point(2);
  point2->AddRawFeature(1);
  point2->AddRawFeature(-0.5);
  point2->AddRawFeature(123);
  point2->AddRawFeature(3);
  Point *point3 = new Point(2);
  point3->AddRawFeature(-1);
  point3->AddRawFeature(0.0);
  point3->AddRawFeature(40);
  point3->AddRawFeature(-3);
  Point *point4 = new Point(2);
  point4->AddRawFeature(-1);
  point4->AddRawFeature(0.1);
  point4->AddRawFeature(-40);
  point4->AddRawFeature(1);
  Sample sample;
  sample.push_back(point1);
  sample.push_back(point2);
  sample.push_back(point3);
  sample.push_back(point4);
  sample.push_back(point4);
  feature->ComputeSampleExpectation(sample);
  EXPECT_NEAR(0.6, feature->GetSampleExpectation(), gTolerance);
}

// Tests computing and getting sample expectation of a monomial feature
TEST(FeatureTest, TestComputeGetSampleExpectationOfMonomialFeature) {
  int mon[3] = {3, 1, 2};
  std::vector<int> monomial(mon, mon + 3);
  MonomialFeature *feature = new MonomialFeature(monomial);
  EXPECT_TRUE(isnan(feature->GetSampleExpectation()));
  Point *point = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Sample sample;
  point->AddRawFeature(-1);
  point->AddRawFeature(0.5);
  point->AddRawFeature(3);
  point2->AddRawFeature(-1);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(2);
  point3->AddRawFeature(1);
  point3->AddRawFeature(1.5);
  point3->AddRawFeature(1);
  sample.push_back(point);
  sample.push_back(point2);
  sample.push_back(point3);
  sample.push_back(point2);
  feature->ComputeSampleExpectation(sample);
  EXPECT_NEAR(-2.75, feature->GetSampleExpectation(), gTolerance);
}


// Tests computing and getting unnormalized expectation of a raw feature
TEST(FeatureTest, TestGetComputePopulationExpectationOfRawFeature) {
  RawFeature *feature = new RawFeature(1);
  EXPECT_TRUE(isnan(feature->GetUnnormalizedPopulationExpectation()));
  Point *point = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Space *space = new Space();
  point->AddRawFeature(-1);
  point->AddRawFeature(0.5);
  point->AddRawFeature(123);
  point2->AddRawFeature(0.0);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(123);
  point3->AddRawFeature(1);
  point3->AddRawFeature(1.5);
  point3->AddRawFeature(123);
  point->SetProbWeight(2);
  point2->SetProbWeight(100);
  point3->SetProbWeight(2 / 1.5);
  space->AddPoint(*point);
  space->AddPoint(*point2);
  space->AddPoint(*point3);
  space->Finalize();
  feature->ComputeUnnormalizedPopulationExpectation(*space);
  EXPECT_NEAR(103.0, feature->GetUnnormalizedPopulationExpectation(),
	      gTolerance);
}

// Tests computing unnormalized expectation of a product feature
TEST(FeatureTest, TestGetComputePopulationExpectationOfProductFeature) {
  ProductFeature *feature = new ProductFeature(0, 1);
  EXPECT_TRUE(isnan(feature->GetUnnormalizedPopulationExpectation()));
  Point *point = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Space *space = new Space();
  point->AddRawFeature(-1);
  point->AddRawFeature(0.5);
  point->AddRawFeature(123);
  point2->AddRawFeature(0.0);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(123);
  point3->AddRawFeature(1);
  point3->AddRawFeature(1.5);
  point3->AddRawFeature(123);
  point->SetProbWeight(2);
  point2->SetProbWeight(100);
  point3->SetProbWeight(2 / 1.5);
  space->AddPoint(*point);
  space->AddPoint(*point2);
  space->AddPoint(*point3);
  space->Finalize();
  feature->ComputeUnnormalizedPopulationExpectation(*space);
  EXPECT_NEAR(1.0, feature->GetUnnormalizedPopulationExpectation(),
	      gTolerance);
}

// Tests computing unnormalized expectation of a threshold feature
TEST(FeatureTest, TestGetComputePopulationExpectationOfThresholdFeature) {
  ThresholdFeature *feature = new ThresholdFeature(0, 1.1);
  EXPECT_TRUE(isnan(feature->GetUnnormalizedPopulationExpectation()));
  Point *point = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Space *space = new Space();
  point->AddRawFeature(-1);
  point->AddRawFeature(0.5);
  point->AddRawFeature(123);
  point2->AddRawFeature(0.0);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(123);
  point3->AddRawFeature(4);
  point3->AddRawFeature(1.5);
  point3->AddRawFeature(123);
  point->SetProbWeight(2);
  point2->SetProbWeight(2);
  point3->SetProbWeight(5);
  space->AddPoint(*point);
  space->AddPoint(*point2);
  space->AddPoint(*point3);
  space->Finalize();
  feature->ComputeUnnormalizedPopulationExpectation(*space);
  EXPECT_NEAR(5.0, feature->GetUnnormalizedPopulationExpectation(),
	      gTolerance);
}

// Tests computing unnormalized population expectation of a tree feature
TEST(FeatureTest, TestComputeGetPopulationExpectationOfTreeFeature) {
  Node *root = new Node();
  root->SetLeftChild(new Node());
  root->SetRightChild(new Node());
  root->GetLeftChild()->SetLeftChild(new Node());
  root->GetLeftChild()->SetRightChild(new Node());
  root->GetRightChild()->SetLeftChild(new Node());
  root->GetRightChild()->SetRightChild(new Node()); 
  root->SetThreshold(0.0);
  root->SetFeature(1);
  root->GetLeftChild()->SetThreshold(1.0);
  root->GetLeftChild()->SetFeature(2);
  root->GetLeftChild()->GetLeftChild()->SetValue(1);
  root->GetLeftChild()->GetRightChild()->SetValue(0);
  root->GetRightChild()->SetThreshold(-2.0);
  root->GetRightChild()->SetFeature(3);
  root->GetRightChild()->GetLeftChild()->SetValue(0);
  root->GetRightChild()->GetRightChild()->SetValue(1);
  TreeFeature *feature = new TreeFeature(root);
  EXPECT_TRUE(isnan(feature->GetUnnormalizedPopulationExpectation()));
  Point *point1 = new Point(1);
  point1->AddRawFeature(1);
  point1->AddRawFeature(-1.0);
  point1->AddRawFeature(0.0);
  point1->AddRawFeature(123);
  Point *point2 = new Point(2);
  point2->AddRawFeature(1);
  point2->AddRawFeature(-0.5);
  point2->AddRawFeature(123);
  point2->AddRawFeature(3);
  Point *point3 = new Point(2);
  point3->AddRawFeature(-1);
  point3->AddRawFeature(0.0);
  point3->AddRawFeature(40);
  point3->AddRawFeature(-3);
  Point *point4 = new Point(2);
  point4->AddRawFeature(-1);
  point4->AddRawFeature(0.1);
  point4->AddRawFeature(-40);
  point4->AddRawFeature(1);
  point1->SetProbWeight(2.0);
  point2->SetProbWeight(1.0);
  point3->SetProbWeight(4.0);
  point4->SetProbWeight(7.0);
  Space *space = new Space();
  space->AddPoint(*point1);
  space->AddPoint(*point2);
  space->AddPoint(*point3);
  space->AddPoint(*point4);
  space->Finalize();
  feature->ComputeUnnormalizedPopulationExpectation(*space);
  EXPECT_NEAR(9.0, feature->GetUnnormalizedPopulationExpectation(),
	      gTolerance);
}

// Tests computing unnormalized expectation population of a monomial feature
TEST(FeatureTest, TestComputeGetPopulationExpectationOfMonomialFeature) {
  int mon[3] = {3, 1, 2};
  std::vector<int> monomial(mon, mon + 3);
  MonomialFeature *feature = new MonomialFeature(monomial);
  EXPECT_TRUE(isnan(feature->GetSampleExpectation()));
  Point *point1 = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Sample sample;
  point1->AddRawFeature(-1);
  point1->AddRawFeature(0.5);
  point1->AddRawFeature(3);
  point2->AddRawFeature(-1);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(2);
  point3->AddRawFeature(1);
  point3->AddRawFeature(1.5);
  point3->AddRawFeature(1);
  point1->SetProbWeight(2.0);
  point2->SetProbWeight(1.0);
  point3->SetProbWeight(4.0);
  Space *space = new Space();
  space->AddPoint(*point1);
  space->AddPoint(*point2);
  space->AddPoint(*point3);
  space->Finalize();
  feature->ComputeUnnormalizedPopulationExpectation(*space);
  EXPECT_NEAR(-7.0, feature->GetUnnormalizedPopulationExpectation(),
	      gTolerance);
}


// Tests that computition expectations of the tree feature based on
// the points and samples stored in the leaf is correct.
TEST(FeatureTest, TestComputeTreeExpectations) {
  Node *root = new Node();
  root->SetLeftChild(new Node());
  root->SetRightChild(new Node());
  root->GetLeftChild()->SetLeftChild(new Node());
  root->GetLeftChild()->SetRightChild(new Node());
  root->GetRightChild()->SetLeftChild(new Node());
  root->GetRightChild()->SetRightChild(new Node()); 
  root->SetThreshold(0.0);
  root->SetFeature(1);
  root->GetLeftChild()->SetThreshold(1.0);
  root->GetLeftChild()->SetFeature(2);
  root->GetLeftChild()->GetLeftChild()->SetValue(1);
  root->GetLeftChild()->GetRightChild()->SetValue(0);
  root->GetRightChild()->SetThreshold(-2.0);
  root->GetRightChild()->SetFeature(3);
  root->GetRightChild()->GetLeftChild()->SetValue(0);
  root->GetRightChild()->GetRightChild()->SetValue(1);
  TreeFeature *feature = new TreeFeature(root);
  EXPECT_EQ(7, feature->TreeSize());
  EXPECT_TRUE(isnan(feature->GetSampleExpectation()));
  EXPECT_TRUE(isnan(feature->GetUnnormalizedPopulationExpectation()));
  Point *point1 = new Point(1);
  Point *point2 = new Point(2);
  Point *point3 = new Point(3);
  Point *point4 = new Point(4);
  Point *point5 = new Point(5);
  Point *point6 = new Point(6);
  Point *point7 = new Point(7);
  Point *point8 = new Point(8);
  Point *point9 = new Point(9);
  Point *point10 = new Point(10);
  Point *point11 = new Point(11);
  Point *point12 = new Point(12);
  point1->SetProbWeight(2.0);
  point2->SetProbWeight(3.0);
  point3->SetProbWeight(1.0);
  point4->SetProbWeight(30.0);
  point5->SetProbWeight(40.0);
  point6->SetProbWeight(50.0);
  point7->SetProbWeight(30.0);
  point8->SetProbWeight(50.0);
  point9->SetProbWeight(40.0);
  point10->SetProbWeight(3.0);
  point11->SetProbWeight(7.0);
  point12->SetProbWeight(5.0);
  root->GetLeftChild()->GetLeftChild()->AddPoint(point1);
  root->GetLeftChild()->GetLeftChild()->AddPoint(point2);
  root->GetLeftChild()->GetLeftChild()->AddPoint(point3);
  root->GetLeftChild()->GetRightChild()->AddPoint(point4);
  root->GetLeftChild()->GetRightChild()->AddPoint(point5);
  root->GetLeftChild()->GetRightChild()->AddPoint(point6);
  root->GetRightChild()->GetLeftChild()->AddPoint(point7);
  root->GetRightChild()->GetLeftChild()->AddPoint(point8);
  root->GetRightChild()->GetLeftChild()->AddPoint(point9);
  root->GetRightChild()->GetRightChild()->AddPoint(point10);
  root->GetRightChild()->GetRightChild()->AddPoint(point11);
  root->GetRightChild()->GetRightChild()->AddPoint(point12);
  root->GetLeftChild()->GetLeftChild()->AddSample(point1);
  root->GetLeftChild()->GetLeftChild()->AddSample(point2);
  root->GetLeftChild()->GetLeftChild()->AddSample(point2);
  root->GetLeftChild()->GetRightChild()->AddSample(point4);
  root->GetLeftChild()->GetRightChild()->AddSample(point4);
  root->GetRightChild()->GetLeftChild()->AddSample(point7);
  feature->ComputeTreeExpectations();
  EXPECT_NEAR(21.0, feature->GetUnnormalizedPopulationExpectation(),
  	      gTolerance);
  EXPECT_NEAR(0.5, feature->GetSampleExpectation(),
  	      gTolerance);
}

// Tests setting and getting complexities.
TEST(FeatureTest, TestComplexity) {
  RawFeature *feature1 = new RawFeature(1);
  ProductFeature *feature2 = new ProductFeature(0, 0);
  ThresholdFeature *feature3 = new ThresholdFeature(0, 0.0);
  ThresholdFeature *feature4 = new ThresholdFeature(1, 1.1);
  TreeFeature *feature5 = new TreeFeature(new Node());
  TreeFeature *feature6 = new TreeFeature(new Node());
  int mon[2] = {1, 2};
  std::vector<int> monomial(mon, mon + 2);
  MonomialFeature *feature7 = new MonomialFeature(monomial);
  MonomialFeature *feature8 = new MonomialFeature(monomial);
  feature1->SetComplexity(1.0);
  feature2->SetComplexity(2.0);
  feature3->SetComplexity(3.0);
  feature5->SetComplexity(5.0);
  feature6->SetComplexity(6.0);
  feature7->SetComplexity(7.0);
  feature8->SetComplexity(8.0);
  EXPECT_NEAR(1.0, feature1->Complexity(), gTolerance);
  EXPECT_NEAR(2.0, feature2->Complexity(), gTolerance);
  EXPECT_NEAR(3.0, feature3->Complexity(), gTolerance);
  EXPECT_NEAR(3.0, feature4->Complexity(), gTolerance);
  EXPECT_NEAR(5.0, feature5->Complexity(), gTolerance);
  EXPECT_NEAR(6.0, feature6->Complexity(), gTolerance);
  EXPECT_NEAR(7.0, feature7->Complexity(), gTolerance);
  EXPECT_NEAR(8.0, feature8->Complexity(), gTolerance);
}
