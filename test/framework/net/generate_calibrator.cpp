#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "saber/core/tensor_op.h"
#include <dirent.h> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include <unistd.h>  
#include <fcntl.h>
#include <map>
#include "framework/operators/ops.h"
#include <fstream>

#if defined(USE_CUDA)
using Target = NV;
using Target_H = X86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#endif

#ifdef USE_GFLAGS
#include <gflags/gflags.h>

DEFINE_string(data_file, "", "calibrator data file");
DEFINE_string(calibrator_file, "", "calibrator file");
DEFINE_string(model_file, "", "model file");
DEFINE_int32(batch_size, 1, "seq num");
DEFINE_int32(bin_num, 2018, "bin num");
#else
std::string FLAGS_data_file;
std::string FLAGS_calibrator_file;
std::string FLAGS_model_file;
int FLAGS_batch_size = 1;
int FLAGS_bin_num = 2048;
#endif

TEST(NetTest, net_execute_base_test) {
    Graph<Target, AK_FLOAT, Precision::FP32> graph;   
    auto status = graph.load(FLAGS_model_file);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    graph.ResetBatchSize("input_0", FLAGS_batch_size);
    LOG(INFO) << "optimize the graph";
    graph.Optimize();
    // constructs the executer net
    LOG(INFO) << "create net to execute";
    //Context<Target> ctx(0, 0, 0);
    //OpContextPtr<Target> ctx;
    BatchStream<X86, AK_FLOAT> stream(FLAGS_data_file, FLAGS_batch_size);
    Calibrator<Target, AK_FLOAT> calibrator(&stream,
              FLAGS_batch_size, FLAGS_bin_num, 
              FLAGS_calibrator_file);
    Net<Target, AK_FLOAT, Precision::FP32> net_executer(graph,  true, &calibrator);
    
}


int main(int argc, const char** argv){

    Env<Target>::env_init();

    // initial logger
    logger::init(argv[0]);

#ifdef USE_GFLAGS
    google::ParseCommandLineFlags(&argc, &argv, true);
#else 
    LOG(INFO)<< "generate_calibrator usage:";
    LOG(INFO)<< "   $generate_calibrator <data_file> <model_file> <calibrator_file> <batch_size>";
    if(argc < 4) {
        LOG(ERROR) << "You should fill in the variable data_file and model_file  calibrator_file at least.";
        return 0;
    }
    FLAGS_data_file = argv[1];
    FLAGS_model_file = argv[2];
    FLAGS_calibrator_file = argv[3];
    if(argc > 4) {
        FLAGS_batch_size = atoi(argv[4]);
    }
#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]); 
    return 0;
}
