cmake_minimum_required(VERSION 3.10)

project(streamDM VERSION 0.0.1)

set(sourcefiles
streamDM/core/Attribute.cpp
streamDM/core/DenseInstance.cpp
streamDM/core/DiscreteEstimator.cpp
streamDM/core/DoubleVector.cpp
streamDM/core/GaussianEstimator.cpp
streamDM/core/Instance.cpp
streamDM/core/InstanceInformation.cpp
streamDM/core/SparseEstimator.cpp
streamDM/core/SparseInstance.cpp
streamDM/core/SplitCriterion.cpp
streamDM/utils/Utils.cpp
streamDM/utils/CommandLineParser.cpp
streamDM/utils/Configurable.cpp
streamDM/utils/DynamicCreateClass.cpp
streamDM/utils/jsoncpp.cpp
streamDM/utils/LearnerModel.cpp
streamDM/utils/RTTI.cpp
streamDM/streams/Reader.cpp
streamDM/streams/ArffReader.cpp
streamDM/streams/C45Reader.cpp
streamDM/streams/CSVReader.cpp
streamDM/streams/LibSVMReader.cpp
streamDM/evaluation/BasicClassificationEvaluator.cpp
streamDM/evaluation/Evaluator.cpp
streamDM/evaluation/Measures.cpp
streamDM/evaluation/NullEvaluator.cpp
streamDM/learners/Learner.cpp
streamDM/learners/Classifiers/Functions/Logisticregression.cpp
streamDM/learners/Classifiers/Functions/Majorityclass.cpp
streamDM/learners/Classifiers/Functions/Perceptron.cpp
streamDM/learners/Classifiers/Bayes/NaiveBayesStatistics.cpp
streamDM/learners/Classifiers/Bayes/Naivebayes.cpp
streamDM/learners/Classifiers/Bayes/SimpleNaiveBayesStatistics.cpp
streamDM/learners/Classifiers/Meta/Bagging.cpp
streamDM/learners/Classifiers/Bayes/observer/NominalAttributeClassObserver.cpp
streamDM/learners/Classifiers/Bayes/observer/NumericAttributeClassObserver.cpp
streamDM/learners/Classifiers/Trees/ADWIN.cpp
streamDM/learners/Classifiers/Trees/AttributeSplitSuggestion.cpp
streamDM/learners/Classifiers/Trees/HATNode.cpp
streamDM/learners/Classifiers/Trees/HoeffdingAdaptiveTree.cpp
streamDM/learners/Classifiers/Trees/HoeffdingTree.cpp
streamDM/learners/Classifiers/Trees/HTAttributeClassObserver.cpp
streamDM/learners/Classifiers/Trees/HTNode.cpp
streamDM/learners/Classifiers/Trees/InstanceConditionalTest.cpp
streamDM/tasks/EvaluatePrequential.cpp
streamDM/tasks/EvaluateHoldOut.cpp
streamDM/tasks/Task.cpp)

set(include_dirs
streamDM/
streamDM/core/
streamDM/learners/
streamDM/utils/
streamDM/learners/classifiers/Bayes/
streamDM/tasks/
streamDM/learners/classifiers/Trees/
streamDM/learners/Classifiers/Bayes/observer/
streamDM/learners/Classifiers/Meta/
streamDM/learners/Classifiers/Functions/
streamDM/evaluation/
streamDM/streams/
spdlog/include/
)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O3 -DUNIX -g")

add_definitions(${flags})

add_subdirectory(spdlog)
# spdlog_enable_warnings(pearl)

add_library(streamDM STATIC ${sourcefiles})

target_link_libraries(streamDM PUBLIC spdlog::spdlog)
target_include_directories(streamDM PUBLIC ${include_dirs})
