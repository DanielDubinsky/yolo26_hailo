#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "preprocess.hpp"

// Simple Detection struct for ONNX results
struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int cls_id;
    std::string cls_name;
};

// COCO Classes (for visualization)
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

std::string get_cls_name(int id) {
    if (id >= 0 && id < (int)COCO_CLASSES.size()) {
        return COCO_CLASSES[id];
    }
    return "Unknown";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [model_path] [conf_thresh]" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "models/yolo26n.onnx";
    float conf_threshold = (argc > 3) ? std::stof(argv[3]) : 0.25f;

    // 1. Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO26");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "Loading model: " << model_path << std::endl;
    Ort::Session session(env, model_path.c_str(), session_options);

    // 2. Preprocess Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not read image " << image_path << std::endl;
        return 1;
    }
    
    int orig_h = img.rows;
    int orig_w = img.cols;

    // Use letterbox preprocessing
    LetterboxInfo lb_info = letterbox_image(img, 640, 640);
    cv::Mat resized = lb_info.image;
    
    // Normalize to [0, 1] and HWC -> CHW
    // OpenCV is BGR, we need RGB for YOLO models usually (unless trained on BGR, but standard is RGB)
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    // Create input tensor
    std::vector<float> input_tensor_values(1 * 3 * 640 * 640);
    
    // HWC to CHW
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 640; h++) {
            for (int w = 0; w < 640; w++) {
                input_tensor_values[c * 640 * 640 + h * 640 + w] = resized.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }

    // 3. Run Inference
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"}; // Standard YOLO export name
    const char* output_names[] = {"output0"}; // Standard YOLO export name

    // If inputs/outputs are dynamic, we should query generic names, but hardcoding for now based on known model.
    // Pro-tip: Verify with session.GetInputName/GetOutputName if needed.

    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_ms = end_time - start_time;
    
    std::cout << "Inference time: " << inference_ms.count() << " ms" << std::endl;

    // 4. Postprocess
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    // Output shape: [1, 300, 6] -> 1800 floats
    
    std::vector<Detection> detections;
    

    // 300 detections
    for (int i = 0; i < 300; i++) {
        float* det = floatarr + (i * 6);
        // [x1, y1, x2, y2, conf, cls]
        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float conf = det[4];
        float cls = det[5];

        if (conf >= conf_threshold) {
             // Reverse Letterbox: (coord - pad) / scale
             float r_x1 = (x1 - lb_info.pad_w) / lb_info.scale;
             float r_y1 = (y1 - lb_info.pad_h) / lb_info.scale;
             float r_x2 = (x2 - lb_info.pad_w) / lb_info.scale;
             float r_y2 = (y2 - lb_info.pad_h) / lb_info.scale;

             // Clip
             Detection d;
             d.x1 = std::max(0.0f, std::min(r_x1, (float)orig_w));
             d.y1 = std::max(0.0f, std::min(r_y1, (float)orig_h));
             d.x2 = std::max(0.0f, std::min(r_x2, (float)orig_w));
             d.y2 = std::max(0.0f, std::min(r_y2, (float)orig_h));
             d.conf = conf;
             d.cls_id = (int)cls;
             d.cls_name = get_cls_name(d.cls_id);
             detections.push_back(d);
             
             std::cout << "Det: " << d.cls_name << " " << conf << " [" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]" << std::endl;
        }
    }

    std::cout << "Found " << detections.size() << " detections." << std::endl;

    // 5. Draw
    for (const auto& det : detections) {
        cv::rectangle(img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);
        std::string label = det.cls_name + " " + std::to_string(det.conf).substr(0, 4);
        cv::putText(img, label, cv::Point(det.x1, det.y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    std::string output_path = "output_onnx_cpp.jpg";
    cv::imwrite(output_path, img);
    std::cout << "Saved to " << output_path << std::endl;

    return 0;
}
