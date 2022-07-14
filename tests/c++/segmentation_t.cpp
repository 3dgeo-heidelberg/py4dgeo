#include "catch2/catch.hpp"
#include "py4dgeo/segmentation.hpp"
#include "testsetup.hpp"

#include <limits>

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
    -8.75263237,  -12.4498439,  -5.95124875,  -7.85988551,  -6.3420341,
    -8.12835494,  -9.1555907,   -8.99132828,  -8.23779461,  -7.04616777,
    -10.19556631, -7.24568867,  -9.96439281,  -7.82714118,  -9.66504466,
    -9.18525009,  -8.98032933,  -12.49556549, -7.24393337,  -7.43629618,
    -12.44862489, -11.26010483, -14.71118047, -12.66358088, -11.4180224,
    -9.89459227,  -9.25428397,  -10.98106482, -12.24054219, -14.11235437,
    -9.67368203,  -8.89575981,  -9.09613235,  -12.36181265, -10.65832827,
    -12.23607436, -12.05403716, -8.41956718,  -5.18927969,  -3.46317324,
    -5.34923737,  -4.53456195,  -9.19263727,  -7.17482253,  -2.20532421,
    -7.37874326,  -1.53594652,  -5.11036,     -2.87127928,  -2.6068047,
    -5.79774031,  -2.94577185,  -1.80551765,  -5.97307174,  -3.47239943,
    -5.97743071,  -1.37331331,  -5.36262128,  -10.04554594, -14.0063941,
    -12.06269672, -10.97716596, -8.74879272,  -9.3208788,   -10.17981112,
    -9.29126698,  -13.14586401, -8.59697448,  -10.69016884, -10.05068367,
    -14.70860228, -10.17570982, -9.95747217,  -13.48605285, -8.98073346,
    -7.88617972,  -12.61344823, -8.60895198,  0.30669687,   -3.8899,
    -3.17765681,  -2.41072433,  -2.38724579,  -3.01936097,  -5.28568242,
    -5.27564404,  -1.32490158,  -6.90442492,  -0.18194116,  -3.49999513,
    -3.92970908,  -7.39331718,  -2.20850241,  -2.95758654,  -1.05819608,
    -3.44517988,  -3.48499704,  -2.39362274,  -3.29244654,  -3.5994971
  };
  IndexType order1 = 1;
  IndexType order2 = 2;
  IndexType order3 = 3;
  IndexType order4 = 4;
  IndexType order5 = 5;
  IndexType order6 = 6;
  IndexType order7 = 7;
  IndexType order8 = 8;
  IndexType order9 = 9;
  IndexType order10 = 10;
  IndexType order11 = 11;
  IndexType order12 = 12;
  IndexType order21 = 21;
  IndexType order22 = 22;

  std::vector<IndexType> result1 = local_maxima_calculation(subsignal, order1);
  std::vector<IndexType> result2 = local_maxima_calculation(subsignal, order2);
  std::vector<IndexType> result3 = local_maxima_calculation(subsignal, order3);
  std::vector<IndexType> result4 = local_maxima_calculation(subsignal, order4);
  std::vector<IndexType> result5 = local_maxima_calculation(subsignal, order5);
  std::vector<IndexType> result6 = local_maxima_calculation(subsignal, order6);
  std::vector<IndexType> result7 = local_maxima_calculation(subsignal, order7);
  std::vector<IndexType> result8 = local_maxima_calculation(subsignal, order8);
  std::vector<IndexType> result9 = local_maxima_calculation(subsignal, order9);
  std::vector<IndexType> result10 =
    local_maxima_calculation(subsignal, order10);
  std::vector<IndexType> result11 =
    local_maxima_calculation(subsignal, order11);
  std::vector<IndexType> result12 =
    local_maxima_calculation(subsignal, order12);
  std::vector<IndexType> result21 =
    local_maxima_calculation(subsignal, order21);
  std::vector<IndexType> result22 =
    local_maxima_calculation(subsignal, order22);

  std::vector<IndexType> true_result1{ 2,  4,  9,  11, 13, 16, 18, 21,
                                       26, 31, 34, 39, 41, 44, 46, 49,
                                       52, 54, 56, 62, 65, 67, 69, 72,
                                       75, 78, 82, 86, 88, 92, 94, 97 };
  std::vector<IndexType> true_result2{ 2,  9,  18, 26, 31, 39, 46, 49, 52,
                                       56, 62, 67, 75, 78, 82, 88, 94, 97 };
  std::vector<IndexType> true_result3{ 9,  18, 26, 31, 39, 46, 52,
                                       56, 62, 67, 78, 82, 88, 94 };
  std::vector<IndexType> true_result4{ 9,  18, 26, 31, 39, 46,
                                       56, 62, 67, 78, 88, 94 };
  std::vector<IndexType> true_result5{ 18, 31, 46, 56, 67, 78, 88, 94 };
  std::vector<IndexType> true_result6{ 18, 46, 56, 67, 78, 88 };
  std::vector<IndexType> true_result7{ 18, 46, 56, 67, 78, 88 };
  std::vector<IndexType> true_result8{ 18, 46, 56, 78, 88 };
  std::vector<IndexType> true_result9{ 46, 56, 78, 88 };
  std::vector<IndexType> true_result10{ 56, 78 };
  std::vector<IndexType> true_result11{ 56, 78 };
  std::vector<IndexType> true_result12{ 56, 78 };
  std::vector<IndexType> true_result21{ 56, 78 };
  std::vector<IndexType> true_result22{ 78 };

  REQUIRE(result1 == true_result1);
  REQUIRE(result2 == true_result2);
  REQUIRE(result3 == true_result3);
  REQUIRE(result4 == true_result4);
  REQUIRE(result5 == true_result5);
  REQUIRE(result6 == true_result6);
  REQUIRE(result7 == true_result7);
  REQUIRE(result8 == true_result8);
  REQUIRE(result9 == true_result9);
  REQUIRE(result10 == true_result10);
  REQUIRE(result11 == true_result11);
  REQUIRE(result12 == true_result12);
  REQUIRE(result21 == true_result21);
  REQUIRE(result22 == true_result22);
}

TEST_CASE("Calculation cost L1", "[segmentation]")
{

  EigenTimeSeries signal(12);
  signal << -8.75263237, -12.4498439, -5.95124875, -7.85988551, -6.3420341,
    -8.12835494, -9.1555907, -8.99132828, -8.23779461, -7.04616777,
    -10.19556631, -7.24568867;
  IndexType min_size = 12;

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
  IndexType min_size = 12;

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

  REQUIRE_THAT(score1, Catch::Approx(true_value1).epsilon(1e-8));
  REQUIRE_THAT(score2, Catch::Approx(true_value2).epsilon(1e-8));
  REQUIRE_THAT(score3, Catch::Approx(true_value3).epsilon(1e-8));
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
  IndexType penalty1 = 1;
  std::vector<double> score1;
  score1 = fit_change_point_detection(signal, window_width1, jump1, min_size1);

  std::vector<IndexType> changepoints1;
  changepoints1 = predict_change_point_detection(
    signal, score1, window_width1, jump1, min_size1, penalty1);
  std::vector<IndexType> true_values1{ 20, 41, 61, 80, 100 };

  IndexType min_size2 = 6;
  IndexType jump2 = 6;
  IndexType window_width2 = 12;
  IndexType penalty2 = 1;
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
  IndexType penalty1 = 1;

  ChangePointDetectionData data{
    signal, window_width1, min_size1, jump1, penalty1
  };
  std::vector<IndexType> cpp = change_point_detection(data);
  std::vector<IndexType> true_values1{ 20, 41, 61, 80, 100 };

  REQUIRE(cpp == true_values1);
}
