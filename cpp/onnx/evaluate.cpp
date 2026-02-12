#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "preprocess.hpp"

// Helper to list files
std::vector<std::string> list_files(const std::string& dir_path) {
    std::vector<std::string> files;
    DIR* dir = opendir(dir_path.c_str());
    if (dir == nullptr) return files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".jpg") {
            files.push_back(filename);
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

// Convert model class index to COCO category ID
// Note: This mapping depends on the model training.
// Standard YOLO models output class index 0-79.
// We need to map 0->1 (Person), etc.
// Using the same map as in python script.
const std::vector<int> COCO_IDS_MAP = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
};

int get_coco_id(int cls_idx) {
    if (cls_idx >= 0 && cls_idx < (int)COCO_IDS_MAP.size()) {
        return COCO_IDS_MAP[cls_idx];
    }
    return cls_idx + 1; // Fallback
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <images_dir> <output_json> [limit]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string images_dir = argv[2];
    std::string output_json = argv[3];
    int limit = (argc > 4) ? std::stoi(argv[4]) : 0;

    // 1. Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO26_Eval");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "Loading model: " << model_path << std::endl;
    Ort::Session session(env, model_path.c_str(), session_options);

    // 2. List images
    std::vector<std::string> image_files = list_files(images_dir);
    if (limit > 0 && limit < (int)image_files.size()) {
        image_files.resize(limit);
    }
    std::cout << "Running evaluation on " << image_files.size() << " images..." << std::endl;

    std::ofstream json_file(output_json);
    json_file << "[";

    bool first = true;
    int processed = 0;

    // Prepare I/O buffers once
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    size_t input_tensor_size = 1 * 3 * 640 * 640;
    std::vector<float> input_tensor_values(input_tensor_size);
    
    const char* input_names[] = {"images"}; 
    const char* output_names[] = {"output0"};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());


    for (const auto& filename : image_files) {
        std::string full_path = images_dir + "/" + filename;
        
        // Extract Image ID from filename (assuming 000000xxxxxx.jpg)
        int image_id = 0;
        try {
            size_t dot_pos = filename.find_last_of(".");
            std::string name_no_ext = (dot_pos == std::string::npos) ? filename : filename.substr(0, dot_pos);
            image_id = std::stoi(name_no_ext);
        } catch (...) {
            std::cerr << "Warning: Could not parse image ID from " << filename << std::endl;
            continue;
        }

        cv::Mat img = cv::imread(full_path);
        if (img.empty()) continue;

        int orig_h = img.rows;
        int orig_w = img.cols;

        // Preprocess
        LetterboxInfo lb_info = letterbox_image(img, 640, 640);
        cv::Mat resized = lb_info.image;
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        // HWC -> CHW and Normalize
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 640; h++) {
                for (int w = 0; w < 640; w++) {
                    input_tensor_values[c * 640 * 640 + h * 640 + w] = resized.at<cv::Vec3b>(h, w)[c] / 255.0f;
                }
            }
        }

        // Run
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        // [1, 300, 6] -> [x1, y1, x2, y2, conf, cls]

        for (int i = 0; i < 300; i++) {
            float* det = output_data + (i * 6);
            float conf = det[4];
            
            if (conf >= 0.001f) { // Low threshold for eval
                // Reverse Letterbox: (coord - pad) / scale
                float x1 = (det[0] - lb_info.pad_w) / lb_info.scale;
                float y1 = (det[1] - lb_info.pad_h) / lb_info.scale;
                float x2 = (det[2] - lb_info.pad_w) / lb_info.scale;
                float y2 = (det[3] - lb_info.pad_h) / lb_info.scale;

                // Clip
                x1 = std::max(0.0f, std::min(x1, (float)orig_w));
                y1 = std::max(0.0f, std::min(y1, (float)orig_h));
                x2 = std::max(0.0f, std::min(x2, (float)orig_w));
                y2 = std::max(0.0f, std::min(y2, (float)orig_h));

                int cls_idx = (int)det[5];
                int cat_id = get_coco_id(cls_idx);

                float w = x2 - x1;
                float h = y2 - y1;

                if (!first) {
                    json_file << ",";
                }
                first = false;

                json_file << "{\"image_id\":" << image_id 
                          << ",\"category_id\":" << cat_id
                          << ",\"bbox\":[" << x1 << "," << y1 << "," << w << "," << h << "]"
                          << ",\"score\":" << conf << "}\n";
            }
        }
        
        processed++;
        if (processed % 100 == 0) {
            std::cout << "Processed " << processed << "/" << image_files.size() << std::endl;
        }
    }

    json_file << "]";
    json_file.close();

    std::cout << "\nEvaluation complete." << std::endl;
    std::cout << "Results saved to detections_cpp.json" << std::endl;
    return 0;
}
