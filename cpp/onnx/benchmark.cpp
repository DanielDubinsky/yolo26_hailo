#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [iterations]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    int iterations = (argc > 2) ? std::stoi(argv[2]) : 100;

    // 1. Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO26_Benchmark");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "Loading model: " << model_path << std::endl;
    Ort::Session session(env, model_path.c_str(), session_options);

    // 2. Prepare Inputs
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    size_t input_tensor_size = 1 * 3 * 640 * 640;
    std::vector<float> input_tensor_values(input_tensor_size);
    // Fill with random data
    for (size_t i = 0; i < input_tensor_size; i++) {
        input_tensor_values[i] = (float)rand() / RAND_MAX;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"}; 
    const char* output_names[] = {"output0"};

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for(int i=0; i<5; i++) {
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    }

    // Benchmark
    std::cout << "Running benchmark for " << iterations << " iterations..." << std::endl;
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
    }

    // Stats
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "ONNX C++ INFERENCE BENCHMARK SUMMARY" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Avg Inference Time: " << mean << " ms +/- " << stdev << " ms" << std::endl;
    std::cout << "Throughput: " << 1000.0 / mean << " FPS" << std::endl;
    
    return 0;
}
