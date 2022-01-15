#include <iostream>
#include <windows.h>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/opencv.hpp>


constexpr float CONFIDENCE_THRESHOLD = 0.6;
constexpr float NMS_THRESHOLD = 0.7;
constexpr int NUM_CLASSES = 1;

const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};


const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

cv::Mat hwnd2mat(HWND hwnd)
{
    HDC hwindowDC, hwindowCompatibleDC;

    int height, width, srcheight, srcwidth;
    HBITMAP hbwindow;
    cv::Mat src;
    BITMAPINFOHEADER  bi;

    hwindowDC = GetDC(hwnd);
    hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
    SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR); 

    RECT windowsize;    // get the height and width of the screen
    GetClientRect(hwnd, &windowsize);

    srcheight = windowsize.bottom;
    srcwidth = windowsize.right;
    height = windowsize.bottom / 1;  //change this to whatever size you want to resize to
    width = windowsize.right / 1;

    src.create(height, width, CV_8UC4);

    // create a bitmap
    hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
    bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
    bi.biWidth = width;
    bi.biHeight = -height;  //this is the line that makes it draw upside down or not
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    // use the previously created device context with the bitmap
    SelectObject(hwindowCompatibleDC, hbwindow);
    // copy from the window device context to the bitmap device context
    StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
    GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

    // avoid memory leak
    DeleteObject(hbwindow);
    DeleteDC(hwindowCompatibleDC);
    ReleaseDC(hwnd, hwindowDC);

    return src;

}

double euclideanDistance(double x1, double y1, double x2, double y2) {
    double x = x1 - x2;
    double y = y1 - y2;
    double dist;
    dist = std::pow(x, 2) + pow(y, 2);
    dist = std::sqrt(dist);
    return dist;
}

int main(){

    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;
    POINT p;
    bool startAimbot = false;

    HWND hwndDesktop = GetDesktopWindow();
    std::vector<std::string> class_names;
    {
        std::ifstream class_file("D:\\computer_vision\\yolov4\\classes.txt");
        if (!class_file)
        {
            std::cerr << "failed to open classes.txt\n";
            return 0;
        }

        std::string line;
        while (std::getline(class_file, line)) {
            if (line == "person") {
                class_names.push_back(line);
            }
            else {
                break;
            }
        }
    }

    //Video Resolution
    RECT windowsize;    // get the height and width of the screen
    GetClientRect(hwndDesktop, &windowsize);
    int frameWidth = windowsize.right;
    int frameHeight = windowsize.bottom;

    //Video Writer Object
    //cv::VideoWriter vidOut("vidOut.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frameWidth, frameHeight));

    //Tracker
    /*cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
    cv::Rect trackingBox;
    bool trackerInit = false;*/


    //auto net = cv::dnn::readNetFromDarknet("D:\\computer_vision\\yolov4\\yolov4.cfg", "D:\\computer_vision\\yolov4\\yolov4.weights");
    auto net = cv::dnn::readNetFromDarknet("D:\\computer_vision\\yolov4-tiny\\yolov4-tiny.cfg", "D:\\computer_vision\\yolov4-tiny\\yolov4-tiny.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();

    while (cv::waitKey(1) < 1)
    {
        if (GetAsyncKeyState(VK_UP)) {
            startAimbot = true;
        }

        if (GetAsyncKeyState(VK_DOWN)) {
            startAimbot = false;
        }
        //std::cout << trackerInit << std::endl;
        frame = hwnd2mat(hwndDesktop);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2RGB);
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }


        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        net.forward(detections, output_names);
        auto dnn_end = std::chrono::steady_clock::now();

        std::vector<int> indices[NUM_CLASSES];
        std::vector<cv::Rect> boxes[NUM_CLASSES];
        std::vector<float> scores[NUM_CLASSES];

        if (startAimbot==true) {
            for (auto& output : detections)
            {
                const auto num_boxes = output.rows;
                for (int i = 0; i < num_boxes; i++)
                {
                    auto x = output.at<float>(i, 0) * frame.cols;
                    auto y = output.at<float>(i, 1) * frame.rows;
                    auto width = output.at<float>(i, 2) * frame.cols;
                    auto height = output.at<float>(i, 3) * frame.rows;
                    cv::Rect rect(x - width / 2, y - height / 2, width, height);

                    for (int c = 0; c < NUM_CLASSES; c++)
                    {
                        auto confidence = *output.ptr<float>(i, 5 + c);
                        if (confidence >= CONFIDENCE_THRESHOLD)
                        {
                            boxes[c].push_back(rect);
                            scores[c].push_back(confidence);
                        }
                    }
                }
            }

            for (int c = 0; c < NUM_CLASSES; c++)
                cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

            int nearest_target_x;
            int nearest_target_y;
            int nearest_target_width;
            int nearest_target_height;
            double nearest_target_dist = 1000.0;
            for (int c = 0; c < NUM_CLASSES; c++)
            {
                for (size_t i = 0; i < indices[c].size(); ++i)
                {
                    const auto color = colors[c % NUM_COLORS];

                    auto idx = indices[c][i];
                    const auto& rect = boxes[c][idx];
                    GetCursorPos(&p);
                    int curs_x = p.x;
                    int curs_y = p.y;
                    if (rect.width < 150)
                    {
                        if (euclideanDistance(curs_x, curs_y, rect.x, rect.y) < nearest_target_dist) {
                            nearest_target_dist = euclideanDistance(curs_x, curs_y, rect.x, rect.y);
                            nearest_target_x = rect.x;
                            nearest_target_y = rect.y;
                            nearest_target_width = rect.width;
                            nearest_target_height = rect.height;
                        }
                        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                        std::ostringstream label_ss;
                        label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                        auto label = label_ss.str();

                        int baseline;
                        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                        cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                        cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));



                    }
                }
            }

            if (GetCursorPos(&p) && nearest_target_dist < 1000)
            {

                SetCursorPos(nearest_target_x + (nearest_target_width / 2), nearest_target_y + (nearest_target_height / 5));

                if (p.x > nearest_target_x && p.x < (nearest_target_x + nearest_target_width) && p.y > nearest_target_y && p.y < (nearest_target_y + nearest_target_height))
                {
                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                    Sleep(100);
                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                }
            }
        }
        

        auto total_end = std::chrono::steady_clock::now();

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

        cv::imshow("output", frame);
    }

    return 0;
}