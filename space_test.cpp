#include <cmath>
#include "gtest/gtest.h"
#include "space.hpp"
#include "constants.hpp"

// Tests getting id from a point.
TEST(SpaceTest, TestGetId) {
  Point *point = new Point(1);
  Point *point2 = new Point(7);
  Point *point3 = new Point(99);
  EXPECT_EQ(1, point->GetId());
  EXPECT_EQ(7, point2->GetId());
  EXPECT_EQ(99, point3->GetId());
}

// Tests getting, setting raw features as well as getting the total count.
TEST(SpaceTest, TestAddGetCountFeatures) {
  Point *point = new Point(1);
  EXPECT_EQ(0, point->AddRawFeature(0.123));
  EXPECT_EQ(1, point->AddRawFeature(0.777));
  EXPECT_EQ(2, point->AddRawFeature(0.0));
  EXPECT_EQ(3, point->NumRawFeatures());
  EXPECT_NEAR(0.0, point->GetRawFeature(2), gTolerance);
  EXPECT_NEAR(0.777, point->GetRawFeature(1), gTolerance);
  EXPECT_NEAR(0.123, point->GetRawFeature(0), gTolerance);
  EXPECT_TRUE(isnan(point->GetRawFeature(3)));
  EXPECT_EQ(3, point->NumRawFeatures());
}

// Tests that adding raw features fails once point is finalized.
TEST(SpaceTest, TestAddRawFeatureFailsAfterPointIsFinalized) {
  Point *point = new Point(1);
  point->AddRawFeature(0.0);
  point->Finalize();
  EXPECT_EQ(1, point->NumRawFeatures());
  EXPECT_EQ(-1, point->AddRawFeature(0.1));
  EXPECT_EQ(1, point->NumRawFeatures());
}

// Tests getting and setting probabilistic weights.
TEST(SpaceTest, TestGetSetProbWeights) {
  Point *point = new Point(1);
  EXPECT_NEAR(1.0, point->GetProbWeight(), gTolerance);
  point->SetProbWeight(0.5);
  EXPECT_NEAR(0.5, point->GetProbWeight(), gTolerance);
}

// Tests that getting and setting probabilistic weights still works
// after point was finalized. 
TEST(SpaceTest, TestGetSetProbWeightsAfterFinalizingStillWorks) {
  Point *point = new Point(1);
  EXPECT_NEAR(1.0, point->GetProbWeight(), gTolerance);
  point->Finalize();
  point->SetProbWeight(0.5);
  EXPECT_NEAR(0.5, point->GetProbWeight(), gTolerance);
}

// Tests adding and getting point to space
TEST(SpaceTest, TestAddGetPoint) {
  Point *point = new Point(1);
  Point *point2 = new Point(7);
  Point *point3 = new Point(99);
  Space *space = new Space();
  EXPECT_EQ(0, space->AddPoint(*point));
  EXPECT_EQ(1, space->AddPoint(*point2));
  EXPECT_EQ(2, space->AddPoint(*point3));
  Point &p = space->GetPoint(0);
  EXPECT_EQ(1, p.GetId());
  p = space->GetPoint(1);
  EXPECT_EQ(7, p.GetId());
  p = space->GetPoint(2);
  EXPECT_EQ(99, p.GetId());
}

// Tests adding and getting point to space as well as counting them.
TEST(SpaceTest, TestAddCountGetPoint) {
  Point *point = new Point(1);
  Point *point2 = new Point(7);
  Point *point3 = new Point(99);
  Space *space = new Space();
  EXPECT_EQ(0, space->AddPoint(*point));
  EXPECT_EQ(1, space->AddPoint(*point2));
  EXPECT_EQ(2, space->AddPoint(*point3));
  Point &p = space->GetPoint(0);
  EXPECT_EQ(1, p.GetId());
  p = space->GetPoint(1);
  EXPECT_EQ(7, p.GetId());
  p = space->GetPoint(2);
  EXPECT_EQ(99, p.GetId());
  EXPECT_EQ(3, space->NumPoints());
}

// Tests adding and getting point to space fails once space is finalized.
// Tests that features of point in space can not be modified once
// space is finalized.
TEST(SpaceTest, TestAddGetPointFailsAfterSpaceIsFinalized) {
  Point *point = new Point(1);
  Point *point2 = new Point(7);
  Point *point3 = new Point(99);
  Space *space = new Space();
  EXPECT_EQ(0, space->AddPoint(*point));
  EXPECT_EQ(1, space->AddPoint(*point2));
  EXPECT_EQ(2, space->NumPoints());
  space->Finalize();
  EXPECT_EQ(-1, space->AddPoint(*point3));
  EXPECT_EQ(2, space->NumPoints());
  Point &p = space->GetPoint(0);
  EXPECT_EQ(1, p.GetId());
  EXPECT_EQ(-1, p.AddRawFeature(0.0));
  EXPECT_EQ(0, p.NumRawFeatures());
  p = space->GetPoint(1);
  EXPECT_EQ(7, p.GetId());
  EXPECT_EQ(-1, p.AddRawFeature(0.0));
  EXPECT_EQ(0, p.NumRawFeatures());
}


// Test iterating through space.
TEST(SpaceTest, TestBeginEndIteratesThroughSpace) {
  Point *point = new Point(1);
  Point *point2 = new Point(7);
  Point *point3 = new Point(99);
  Space *space = new Space();
  EXPECT_EQ(0, space->AddPoint(*point));
  EXPECT_EQ(1, space->AddPoint(*point2));
  EXPECT_EQ(2, space->AddPoint(*point3));
  int ids[3] = {1, 7, 99};
  int i = 0;
  for (auto &p : *space) {
    EXPECT_EQ(ids[i], p.GetId());
    i++;
  }
}
