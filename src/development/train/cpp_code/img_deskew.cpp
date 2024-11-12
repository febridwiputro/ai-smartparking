#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

vector<Point> shiTomashi(Mat image, bool is_show = false) {
    Mat gray;
    cvtColor(image, gray, COLOR_RGB2GRAY);
    vector<Point> corners;
    vector<Point2f> features;
    goodFeaturesToTrack(gray, features, 4, 0.01, 100);

    if (features.empty()) {
        cout << "No corners found." << endl;
        return corners;
    }

    for (const auto& pt : features) {
        corners.push_back(Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }

    // Sort corners
    sort(corners.begin(), corners.end(), [](const Point& a, const Point& b) {
        return (a.y < b.y) || (a.y == b.y && a.x < b.x);
    });

    cout << "\nThe corner points are...\n";
    Mat im = image.clone();
    for (size_t i = 0; i < corners.size(); ++i) {
        circle(im, corners[i], 3, Scalar(255, 0, 0), -1);
        putText(im, string(1, 'A' + static_cast<char>(i)), corners[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
        cout << char('A' + i) << ": " << corners[i] << endl;
    }

    if (is_show) {
        imshow("Corner Detection: Shi-Tomashi", im);
        waitKey(0);
    }

    return corners;
}

tuple<vector<Point2f>, int, int> getDestinationPoints(const vector<Point>& corners) {
    if (corners.size() < 4) {
        cout << "Insufficient corners for perspective transform." << endl;
        return make_tuple(vector<Point2f>(), 0, 0);
    }

    double w1 = norm(corners[0] - corners[1]);
    double w2 = norm(corners[2] - corners[3]);
    int w = max(static_cast<int>(w1), static_cast<int>(w2));

    double h1 = norm(corners[0] - corners[2]);
    double h2 = norm(corners[1] - corners[3]);
    int h = max(static_cast<int>(h1), static_cast<int>(h2));

    vector<Point2f> destination_corners = { Point2f(0, 0), Point2f(w - 1, 0), Point2f(0, h - 1), Point2f(w - 1, h - 1) };

    cout << "\nThe destination points are: \n";
    for (size_t i = 0; i < destination_corners.size(); ++i) {
        cout << char('A' + i) << "' : " << destination_corners[i] << endl;
    }

    cout << "\nThe approximated height and width of the original image is: \n" << h << ", " << w << endl;
    return make_tuple(destination_corners, h, w);
}

Mat unwarp(Mat img, const vector<Point2f>& src, const vector<Point2f>& dst) {
    Mat H = findHomography(src, dst, RANSAC);
    if (H.empty()) {
        cout << "Failed to calculate homography matrix." << endl;
        return img;  // Return original if homography fails
    }
    Mat un_warped;
    warpPerspective(img, un_warped, H, img.size());
    return un_warped;
}

Mat applyFilter(Mat image, bool is_show = false) {
    Mat gray, filtered;
    cvtColor(image, gray, COLOR_RGB2GRAY);
    Mat kernel = Mat::ones(5, 5, CV_32F) / 15;
    filter2D(gray, filtered, -1, kernel);

    if (is_show) {
        imshow("Filtered Image", filtered);
        waitKey(0);
    }
    return filtered;
}

Mat applyThreshold(Mat filtered, bool is_show = false) {
    Mat thresh;
    threshold(filtered, thresh, 250, 255, THRESH_OTSU);

    if (is_show) {
        imshow("After applying OTSU threshold", thresh);
        waitKey(0);
    }
    return thresh;
}

void showBoundingBoxes(Mat thresh) {
    Mat img_with_boxes = thresh.clone();
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& cntr : contours) {
        Rect boundRect = boundingRect(cntr);
        rectangle(img_with_boxes, boundRect, Scalar(0, 255, 0), 1);
    }

    imshow("Bounding Boxes on Thresholded Image", img_with_boxes);
    waitKey(0);
}

vector<Point> detectCornersFromContour(Mat& canvas, const vector<Point>& cnt) {
    vector<Point> approx_corners;
    approxPolyDP(cnt, approx_corners, 0.02 * arcLength(cnt, true), true);

    for (size_t i = 0; i < approx_corners.size(); ++i) {
        putText(canvas, string(1, 'A' + static_cast<char>(i)), approx_corners[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
    }
    imshow("Corner Points: Douglas-Peucker", canvas);
    waitKey(0);
    return approx_corners;
}

void exampleTwo(string img_path) {
    Mat image = imread(img_path);
    if (image.empty()) {
        cout << "Image not found." << endl;
        return;
    }
    cvtColor(image, image, COLOR_BGR2RGB);

    Mat filtered_image = applyFilter(image);
    Mat threshold_image = applyThreshold(filtered_image);
    showBoundingBoxes(threshold_image);

    vector<vector<Point>> contours;
    findContours(threshold_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        cout << "No contours found." << endl;
        return;
    }

    vector<Point> largest_contour = contours[0];
    vector<Point> corners = detectCornersFromContour(image, largest_contour);

    if (corners.size() < 4) {
        cout << "Not enough corners detected for transformation." << endl;
        return;
    }

    vector<Point2f> destination_points;
    int h, w;
    tie(destination_points, h, w) = getDestinationPoints(corners);

    if (destination_points.empty()) {
        cout << "Unable to define destination points." << endl;
        return;
    }

    Mat un_warped = unwarp(image, vector<Point2f>(corners.begin(), corners.end()), destination_points);
    Mat cropped = un_warped(Rect(0, 0, w, h));

    imshow("Unwarped Image", un_warped);
    imshow("Cropped Plate Image", cropped);
    waitKey(0);
}

int main() {
    string img_path = "path_to_your_image.jpg";
    exampleTwo(img_path);
    return 0;
}
