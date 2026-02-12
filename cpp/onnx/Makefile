CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -I.. $(shell pkg-config --cflags opencv4) -I/usr/include/onnxruntime/
# Link only specific OpenCV modules to avoid pulling in opencv_dnn (which conflicts with onnxruntime)
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
LDFLAGS = $(OPENCV_LIBS) -lonnxruntime

TARGETS = detect_image benchmark evaluate

all: $(TARGETS)

detect_image: detect_image.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

benchmark: benchmark.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

evaluate: evaluate.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

.PHONY: all clean
