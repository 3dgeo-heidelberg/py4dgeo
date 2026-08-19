#include "testsetup.hpp"
#include <py4dgeo/segmentation.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace py4dgeo;

TEST_CASE("DTW distance calculation", "[segmentation]")
{
  // Wikipedia test case: https://de.wikipedia.org/wiki/Dynamic-Time-Warping
  EigenSpatiotemporalArray arr(2, 4);
  arr << 1, 5, 4, 2, 1, 2, 4, 1;

  TimeseriesDistanceFunctionData data{ arr.row(0), arr.row(1), 0.0, 0.0 };
  auto dist = dtw_distance(data);

  REQUIRE(dist > 0);
  REQUIRE(std::abs(dist - 3) < 1e-8);
}

TEST_CASE("DTW distance with NaN Values", "[segmentation]")
{
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EigenSpatiotemporalArray arr(2, 6);
  arr << nan, 1, 42, 5, 4, 2, 42, 1, nan, 2, 4, 1;
  TimeseriesDistanceFunctionData data{ arr.row(0), arr.row(1), 0.0, 0.0 };
  auto dist = dtw_distance(data);
  REQUIRE(std::abs(dist - 3) < 1e-8);
}

TEST_CASE("DTW distance with all NaN Values", "[segmentation]")
{
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EigenSpatiotemporalArray arr(2, 2);
  arr << nan, nan, nan, nan;
  TimeseriesDistanceFunctionData data{ arr.row(0), arr.row(1), 0.0, 0.0 };
  auto dist = dtw_distance(data);
  REQUIRE(std::isnan(dist));
}

TEST_CASE("Normalized DTW Distances", "[segmentation]")
{
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EigenSpatiotemporalArray arr(2, 6);
  arr << nan, 1, 42, 5, 4, 2, 42, 1, nan, 2, 4, 1;
  TimeseriesDistanceFunctionData data{ arr.row(0), arr.row(1), 0.0, 0.0 };
  auto dist = normalized_dtw_distance(data);
  REQUIRE(dist >= 0.0);
  REQUIRE(dist <= 1.0);
}

TEST_CASE("Median_calculation for 14 values including negative",
          "[segmentation]")
{
  std::vector<double> subsignal{ 9.0, 1, 9, 2,   8,  3,   74,
                                 4,   6, 3, -34, 56, -25, -7 };
  double mean = median_calculation(subsignal);
  REQUIRE(std::abs(mean - 3.5) < 1e-8);
}

TEST_CASE("Median_calculation for 13 values including negative",
          "[segmentation]")
{
  std::vector<double> subsignal{ 9, 1, 9, 2, 8, 74, 6, 0, 3, -34, 56, -25, -7 };
  double mean = median_calculation(subsignal);
  REQUIRE(std::abs(mean - 3.0) < 1e-8);
}

TEST_CASE("Local maxima calculation", "[segmentation]")
{
  std::vector<double> subsignal{
    6.99863915e+01, 9.06408359e+01, 8.13410671e+01, 6.77831025e+01,
    5.31590789e+01, 3.82774617e+01, 2.11088802e+01, 1.86900940e+00,
    1.86900940e+00, 2.97618622e+00, 4.06579088e+00, 3.53954452e+00,
    1.46886774e+00, 1.75437980e-01, 6.02897540e-01, 1.87717908e+00,
    3.61690920e-01, 3.61690920e-01, 7.10542736e-15, 1.88687618e+00,
    3.71578220e+00, 4.92274156e+00, 3.39755630e+00, 1.56865028e+00,
    2.31733020e+00, 1.11751678e+01, 1.70051315e+01, 2.54870722e+01,
    3.44543235e+01, 4.58417830e+01, 5.90938513e+01, 5.47172168e+01,
    5.54368480e+01, 4.43110233e+01, 3.36282954e+01, 2.17050262e+01,
    1.62525474e+01, 9.20163784e+00, 6.17964200e+00, 0.00000000e+00,
    2.34560410e+00, 1.26792540e+01, 2.53748679e+01, 3.53479526e+01,
    5.23087256e+01, 5.24978726e+01, 4.90522323e+01, 4.10721229e+01,
    3.92949571e+01, 2.80161810e+01, 1.79132547e+01, 4.38646398e+00,
    7.15155678e+00, 9.98924200e+00, 1.52767758e+01, 2.66401552e+01,
    3.33803080e+01, 4.60923062e+01, 6.25816178e+01, 7.87648505e+01,
    8.26367941e+01, 7.21430804e+01, 5.81567697e+01, 4.67933903e+01,
    2.88612965e+01, 1.40530121e+01, 2.94313726e+00, -1.42108547e-14,
    4.36233604e+00, 1.07040682e+01, 1.57973338e+01, 2.20795663e+01,
    2.72791308e+01, 3.48844576e+01, 3.37712687e+01, 3.35675227e+01
  };
  IndexType order1 = 1;
  IndexType order2 = 2;
  IndexType order3 = 3;
  IndexType order4 = 4;
  IndexType order12 = 12;
  IndexType order13 = 13;
  IndexType order16 = 16;
  IndexType order17 = 17;
  IndexType order26 = 26;
  IndexType order27 = 27;

  std::vector<IndexType> result1 = local_maxima_calculation(subsignal, order1);
  std::vector<IndexType> result2 = local_maxima_calculation(subsignal, order2);
  std::vector<IndexType> result3 = local_maxima_calculation(subsignal, order3);
  std::vector<IndexType> result4 = local_maxima_calculation(subsignal, order4);
  std::vector<IndexType> result12 =
    local_maxima_calculation(subsignal, order12);
  std::vector<IndexType> result13 =
    local_maxima_calculation(subsignal, order13);
  std::vector<IndexType> result16 =
    local_maxima_calculation(subsignal, order16);
  std::vector<IndexType> result17 =
    local_maxima_calculation(subsignal, order17);
  std::vector<IndexType> result26 =
    local_maxima_calculation(subsignal, order26);
  std::vector<IndexType> result27 =
    local_maxima_calculation(subsignal, order27);

  std::vector<IndexType> true_result1{ 1, 10, 15, 21, 30, 32, 45, 60, 73 };
  std::vector<IndexType> true_result2{ 1, 10, 15, 21, 30, 45, 60, 73 };
  std::vector<IndexType> true_result3{ 1, 10, 15, 21, 30, 45, 60 };
  std::vector<IndexType> true_result4{ 1, 30, 45, 60 };
  std::vector<IndexType> true_result12{ 1, 30, 45, 60 };
  std::vector<IndexType> true_result13{ 1, 30, 60 };
  std::vector<IndexType> true_result16{ 1, 30, 60 };
  std::vector<IndexType> true_result17{ 1, 30 };
  std::vector<IndexType> true_result26{ 1, 30 };
  std::vector<IndexType> true_result27{ 1 };

  REQUIRE(result1 == true_result1);
  REQUIRE(result2 == true_result2);
  REQUIRE(result3 == true_result3);
  REQUIRE(result4 == true_result4);
  REQUIRE(result12 == true_result12);
  REQUIRE(result13 == true_result13);
  REQUIRE(result16 == true_result16);
  REQUIRE(result17 == true_result17);
  REQUIRE(result26 == true_result26);
  REQUIRE(result27 == true_result27);
}

TEST_CASE("Calculation cost L1", "[segmentation]")
{

  EigenTimeSeries signal(12);
  signal << -8.75263237, -12.4498439, -5.95124875, -7.85988551, -6.3420341,
    -8.12835494, -9.1555907, -8.99132828, -8.23779461, -7.04616777,
    -10.19556631, -7.24568867;
  IndexType min_size = 2;

  int start1 = 0;
  int end1 = 12;
  double result1 = cost_L1_error(signal, start1, end1, min_size);

  int start2 = 4;
  int end2 = 10;
  double result2 = cost_L1_error(signal, start2, end2, min_size);

  int start3 = 1;
  int end3 = 1;
  double result3 = cost_L1_error(signal, start3, end3, min_size);

  REQUIRE(std::abs(15.209376429999999 - result1) < 1e-8);
  REQUIRE(std::abs(4.868156779999999 - result2) < 1e-8);
  REQUIRE(std::abs(0.0 - result3) < 1e-8);
}

TEST_CASE("Calculation sum of costs", "[segmentation]")
{
  EigenTimeSeries signal(12);
  signal << 1.89491096e+00, 0.00000000e+00, 3.87843620e-01, 0.00000000e+00,
    1.48506944e+00, 3.55271368e-15, -3.55271368e-15, 4.14970948e+00,
    3.19012034e+00, 7.65566496e+00, 7.65566496e+00, 4.14970948e+00;
  IndexType min_size = 2;

  std::vector<IndexType> bkps1{ 12 };
  std::vector<IndexType> bkps2{ 0, 12 };
  std::vector<IndexType> bkps3{ 12, 12 };
  std::vector<IndexType> bkps4{ 4, 10, 12 };
  std::vector<IndexType> bkps5{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 };

  double error1 = sum_of_costs(signal, bkps1, min_size);
  double error2 = sum_of_costs(signal, bkps2, min_size);
  double error3 = sum_of_costs(signal, bkps3, min_size);
  double error4 = sum_of_costs(signal, bkps4, min_size);
  double error5 = sum_of_costs(signal, bkps5, min_size);

  REQUIRE(std::abs(26.822867119999998 - error1) < 1e-8);
  REQUIRE(std::abs(26.822867119999998 - error2) < 1e-8);
  REQUIRE(std::abs(26.822867119999998 - error3) < 1e-8);
  REQUIRE(std::abs(19.2991354 - error4) < 1e-8);
  REQUIRE(std::abs(3.50595548 - error5) < 1e-8);
}

TEST_CASE("Fit change point detection", "[segmentation]")
{
  EigenTimeSeries signal(20);
  signal << 5.89091385, 3.09297892, 7.54350943, 5.70123322, 11.45957527,
    6.65151675, 6.66711702, 4.32121252, 5.36729293, 6.3855735, 6.15007257,
    4.60662807, 2.93636227, 5.14934393, 4.08923314, 8.09501888, 7.34253289,
    4.77373406, 4.23434058, 6.46301976;

  IndexType min_size1 = 3;
  IndexType jump1 = 1;
  IndexType window_width1 = 6;

  std::vector<double> score1;
  score1 = fit_change_point_detection(signal, window_width1, jump1, min_size1);
  std::vector<double> true_value1{ 0.7606029,  0.9658838,  0.89199268,
                                   1.28422382, 1.83171059, 0.50144418,
                                   0.78277964, 0.76066486, 1.9793425,
                                   2.06083943, 0.54271586, 2.73590482,
                                   3.25329975, 0.37560987 };

  IndexType min_size2 = 10;
  IndexType jump2 = 1;
  IndexType window_width2 = 20;

  std::vector<double> score2;
  score2 = fit_change_point_detection(signal, window_width2, jump2, min_size2);
  std::vector<double> true_value2{};

  IndexType min_size3 = 2;
  IndexType jump3 = 3;
  IndexType window_width3 = 4;

  std::vector<double> score3;
  score3 = fit_change_point_detection(signal, window_width3, jump3, min_size3);
  std::vector<double> true_value3{ 0.0, 0.0, 1.56555928, 0.0, 4.38637792 };

  REQUIRE_THAT(score1, Catch::Matchers::Approx(true_value1).epsilon(1e-8));
  REQUIRE_THAT(score2, Catch::Matchers::Approx(true_value2).epsilon(1e-8));
  REQUIRE_THAT(score3, Catch::Matchers::Approx(true_value3).epsilon(1e-8));
}

TEST_CASE("Predict change point detection", "[segmentation]")
{

  EigenTimeSeries signal(100);
  signal << -9.30567119, -6.50542506, -5.25334064, -3.95708071, -10.62295044,
    -6.00679331, -7.10972198, -4.75825001, -6.7100845, -8.11943878, -9.56607421,
    -4.94672353, -6.67989247, -4.65803801, -7.37845623, -7.48818285,
    -5.78868842, -5.63853894, -6.82157223, -4.36439707, -10.23610407,
    -7.17848293, -10.35211983, -11.3509054, -11.61486292, -9.35727084,
    -10.34027666, -10.86800044, -9.93429977, -8.48897875, -15.3695928,
    -14.26784041, -9.31700749, -10.71438333, -6.79820964, -9.38362261,
    -10.55322992, -10.752245, -13.40571741, -14.18765576, -9.54996564,
    -19.35102448, -20.7635553, -19.26623954, -20.00548551, -17.99023991,
    -18.37537823, -17.53357472, -23.12740104, -17.55591524, -20.86915052,
    -17.75074536, -20.28542954, -20.5310157, -18.73412243, -19.07024164,
    -20.43017388, -21.11249791, -18.45269281, -18.07089436, -19.96608851,
    -26.63058919, -28.97962105, -25.47049917, -26.02732842, -25.42334059,
    -24.96684487, -22.69612178, -24.85381947, -25.34017963, -27.46753899,
    -24.05766122, -29.416168, -19.5100961, -25.98309514, -26.24838995,
    -25.40961916, -23.21292358, -26.63478854, -29.04883697, -16.35986185,
    -12.87413666, -16.2863866, -15.55762212, -21.88964465, -17.68327092,
    -16.60648862, -13.49961018, -15.97483962, -17.23104101, -18.17963806,
    -16.19978101, -17.62239823, -17.14176606, -14.73581351, -14.62187256,
    -20.06316631, -14.96743604, -19.01623592, -15.53150285;

  IndexType min_size1 = 12;
  IndexType jump1 = 1;
  IndexType window_width1 = 24;
  double penalty1 = 1;
  std::vector<double> score1;
  score1 = fit_change_point_detection(signal, window_width1, jump1, min_size1);

  std::vector<IndexType> changepoints1;
  changepoints1 = predict_change_point_detection(
    signal, score1, window_width1, jump1, min_size1, penalty1);
  std::vector<IndexType> true_values1{ 20, 41, 61, 80, 100 };

  IndexType min_size2 = 6;
  IndexType jump2 = 6;
  IndexType window_width2 = 12;
  double penalty2 = 1;
  std::vector<double> score2;
  score2 = fit_change_point_detection(signal, window_width2, jump2, min_size2);

  std::vector<IndexType> changepoints2;
  changepoints2 = predict_change_point_detection(
    signal, score2, window_width2, jump2, min_size2, penalty2);
  std::vector<IndexType> true_values2{ 18, 42, 60, 78, 100 };

  REQUIRE(changepoints1 == true_values1);
  REQUIRE(changepoints2 == true_values2);
}

TEST_CASE("Fit predict change point detection", "[segmentation]")
{
  EigenTimeSeries signal(100);
  signal << -9.30567119, -6.50542506, -5.25334064, -3.95708071, -10.62295044,
    -6.00679331, -7.10972198, -4.75825001, -6.7100845, -8.11943878, -9.56607421,
    -4.94672353, -6.67989247, -4.65803801, -7.37845623, -7.48818285,
    -5.78868842, -5.63853894, -6.82157223, -4.36439707, -10.23610407,
    -7.17848293, -10.35211983, -11.3509054, -11.61486292, -9.35727084,
    -10.34027666, -10.86800044, -9.93429977, -8.48897875, -15.3695928,
    -14.26784041, -9.31700749, -10.71438333, -6.79820964, -9.38362261,
    -10.55322992, -10.752245, -13.40571741, -14.18765576, -9.54996564,
    -19.35102448, -20.7635553, -19.26623954, -20.00548551, -17.99023991,
    -18.37537823, -17.53357472, -23.12740104, -17.55591524, -20.86915052,
    -17.75074536, -20.28542954, -20.5310157, -18.73412243, -19.07024164,
    -20.43017388, -21.11249791, -18.45269281, -18.07089436, -19.96608851,
    -26.63058919, -28.97962105, -25.47049917, -26.02732842, -25.42334059,
    -24.96684487, -22.69612178, -24.85381947, -25.34017963, -27.46753899,
    -24.05766122, -29.416168, -19.5100961, -25.98309514, -26.24838995,
    -25.40961916, -23.21292358, -26.63478854, -29.04883697, -16.35986185,
    -12.87413666, -16.2863866, -15.55762212, -21.88964465, -17.68327092,
    -16.60648862, -13.49961018, -15.97483962, -17.23104101, -18.17963806,
    -16.19978101, -17.62239823, -17.14176606, -14.73581351, -14.62187256,
    -20.06316631, -14.96743604, -19.01623592, -15.53150285;

  IndexType min_size1 = 12;
  IndexType jump1 = 1;
  IndexType window_width1 = 24;
  double penalty1 = 1;

  ChangePointDetectionData data{
    signal, window_width1, min_size1, jump1, penalty1
  };
  std::vector<IndexType> cpp = change_point_detection(data);
  std::vector<IndexType> true_values1{ 20, 41, 61, 80, 100 };

  REQUIRE(cpp == true_values1);
}
