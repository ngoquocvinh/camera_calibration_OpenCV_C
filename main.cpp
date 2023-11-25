
/*
  This program captures video from a camera, detects a chessboard pattern, calibrates the camera,
  undistorts an image, and then processes a video stream to find and measure distances between objects.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// Function to compute the distance in pixels between corners of a chessboard pattern
double computeDistancePixels(const vector<Point2f> &corners);

// Function to compare contour areas for sorting
bool compareContourAreas(vector<Point> contour1, vector<Point> contour2);

// Function to compute the scale factor based on the chessboard pattern
double computeScaleFactor(const Size &patternSize, const float squareSize, const vector<Point2f> &corners);

// Function to compute the distance in real-world units between two points
double computeDistanceRealUnits(const Point2f &p1, const Point2f &p2, double scaleFactor);

// Function to process the video stream and draw boundaries around detected objects
void process_and_draw_boundaries(Mat &frame, Mat &threshold_output, vector<Point2f> &centroids,
                                 const Mat &cameraMatrix, const vector<double> &distCoeffs, const Size &patternSize, const float squareSize, const vector<Point2f> &imagePoints);

int main()
{

    // Images and parameters for chessboard pattern calibration
    Mat image, otsu;
    Size patternSize(9, 6);
    int samples = 2;
    int collected_samples = 0;

    // 3D object points for chessboard pattern calibration
    vector<Point3f> objectPoints;
    float squareSize = 22.0;

    // Generate 3D object points based on the chessboard pattern
    for (int i = 0; i < patternSize.height; i++)
    {
        for (int j = 0; j < patternSize.width; j++)
        {
            objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    // Vectors to store object points and image points for calibration
    vector<vector<Point3f>> objectPointsArray;
    vector<vector<Point2f>> imagePointsArray;

    const string videoStreamAddress = "http://192.168.1.4:4747/video";
    VideoCapture vcap;
    // Open video stream
    if (!vcap.open(videoStreamAddress))
    {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    // Create a window for displaying the chessboard pattern
    namedWindow("Chessboard", WINDOW_NORMAL);

    // Collect chessboard pattern samples
    while (collected_samples < samples)
    {
        vcap.read(image);
        imshow("Chessboard", image);

        int key = waitKey(0);

        if (key == 27)
        {
            break;
        }

        if (key == ' ')
        {
            vector<Point2f> corners;
            bool found = findChessboardCorners(image, patternSize, corners);

            if (found)
            {
                drawChessboardCorners(image, patternSize, corners, found);
                imshow("Chessboard", image);
                collected_samples++;

                imagePointsArray.push_back(corners);
                objectPointsArray.push_back(objectPoints);

                cout << "Collected samples: " << collected_samples << endl;
            }
        }
    }

    // Calibrate the camera
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    vector<double> distCoeffs;
    vector<Mat> rvecs, tvecs;

    calibrateCamera(objectPointsArray, imagePointsArray, image.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

    // Display camera calibration results
    cout << "Camera matrix: " << format(cameraMatrix, Formatter::FMT_CSV) << endl;

    Mat distCoeffsMat = Mat(distCoeffs);
    cout << "Distortion coefficients: " << format(distCoeffsMat, Formatter::FMT_CSV) << endl;

    // Undistort the captured image
    Mat undistortedImage;
    undistort(image, undistortedImage, cameraMatrix, distCoeffs);

    // Find and mark two random points in the undistorted image
    Point2f randomPoint1, randomPoint2;

    // Ensure there are at least two points in the list
    if (imagePointsArray[0].size() >= 2)
    {
        // Generate two random indices
        int index1 = rand() % imagePointsArray[0].size();
        int index2;

        do
        {
            index2 = rand() % imagePointsArray[0].size();
        } while (index2 == index1);

        // Retrieve the corresponding points
        randomPoint1 = imagePointsArray[0][index1];
        randomPoint2 = imagePointsArray[0][index2];

        // Mark the randomly selected points
        circle(undistortedImage, randomPoint1, 5, Scalar(0, 255, 0), -1);
        circle(undistortedImage, randomPoint2, 5, Scalar(0, 255, 0), -1);

        // Compute and display distance in pixels and real-world units
        double distancePixels = norm(randomPoint1 - randomPoint2);
        double distanceMM = computeDistanceRealUnits(randomPoint1, randomPoint2, computeScaleFactor(patternSize, squareSize, imagePointsArray[0]));

        putText(undistortedImage, "Distance: " + to_string(distancePixels) + " pixels", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        putText(undistortedImage, "Distance: " + to_string(distanceMM) + " mm", Point(10, 60),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

        // Display the original and annotated images
        imshow("Original Image", image);
        imshow("Annotated Image", undistortedImage);
        waitKey(0);
    }
    else
    {
        cout << "Not enough points in the list to choose randomly." << endl;
        return -1;
    }

    // Process video stream for object detection and distance measurement
    vector<Point2f> centroids;

    for (;;)
    {
        if (!vcap.read(image))
        {
            cerr << "No frame" << endl;
            waitKey();
            break;
        }

        process_and_draw_boundaries(image, otsu, centroids, cameraMatrix, distCoeffs, patternSize, squareSize, imagePointsArray[0]);

        if (waitKey(1) >= 0)
        {
            break;
        }
    }

    // Release video capture and destroy windows
    vcap.release();
    destroyAllWindows();

    return 0;
}

// Function to process the video stream and draw boundaries around detected objects
void process_and_draw_boundaries(Mat &frame, Mat &threshold_output, vector<Point2f> &centroids,
                                 const Mat &cameraMatrix, const vector<double> &distCoeffs, const Size &patternSize, const float squareSize, const vector<Point2f> &detectedPoints)
{
    // Convert the frame to HSV color space
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // Threshold the frame to detect blue objects
    Mat blueMask;
    inRange(hsv, Scalar(100, 50, 50), Scalar(130, 255, 255), blueMask);

    // Equalize the histogram of the saturation channel
    vector<Mat> channels;
    split(hsv, channels);
    equalizeHist(channels[1], channels[1]);
    merge(channels, hsv);

    // Apply Gaussian blur to the frame
    Mat blurred;
    GaussianBlur(hsv, blurred, Size(5, 5), 0);

    // Convert the blurred frame to grayscale
    Mat gray;
    cvtColor(blurred, gray, COLOR_BGR2GRAY);

    // Bitwise AND operation to combine gray and blue masks
    bitwise_and(gray, blueMask, threshold_output);

    // Find contours in the thresholded image
    vector<vector<Point>> contours;
    findContours(threshold_output, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Sort contours based on area
    sort(contours.begin(), contours.end(), compareContourAreas);

    // Clear the centroids vector
    centroids.clear();

    // Draw contours and mark centroids for up to two largest contours
    for (size_t i = 0; i < min(contours.size(), size_t(2)); ++i)
    {
        Moments mu = moments(contours[i]);
        Point centroid(static_cast<int>(mu.m10 / mu.m00), static_cast<int>(mu.m01 / mu.m00));
        circle(frame, centroid, 5, Scalar(0, 0, 255), -1);

        centroids.push_back(centroid);
    }

    // If there are at least two centroids, compute and display distance information
    if (centroids.size() >= 2)
    {
        double distance_px = norm(centroids[0] - centroids[1]);

        double distance_mm = computeDistanceRealUnits(centroids[0], centroids[1], computeScaleFactor(patternSize, squareSize, detectedPoints));

        // Display distance information on the frame
        ostringstream textMm;
        textMm << "Distance: " << distance_mm << " mm";
        putText(frame, textMm.str(), Point(10, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);

        ostringstream textPx;
        textPx << "Distance: " << distance_px << " pixels";
        putText(frame, textPx.str(), Point(10, 90), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
    }

    // Display the frame with contours and centroids
    imshow("Video with Contours and Centroid", frame);
    waitKey(1);
}

// Function to compute the distance in pixels between corners of a chessboard pattern
double computeDistancePixels(const vector<Point2f> &corners)
{
    double maxDistance = -1;
    for (int j = 0; j < corners.size(); j++)
    {
        for (int k = j + 1; k < corners.size(); k++)
        {
            double distance = norm(corners[j] - corners[k]);
            if (distance > maxDistance)
            {
                maxDistance = distance;
            }
        }
    }
    return maxDistance;
}

// Function to compare contour areas for sorting
bool compareContourAreas(vector<Point> contour1, vector<Point> contour2)
{
    double i = fabs(contourArea(Mat(contour1)));
    double j = fabs(contourArea(Mat(contour2)));
    return (i > j);
}

// Function to compute the scale factor based on the chessboard pattern
double computeScaleFactor(const Size &patternSize, const float squareSize, const vector<Point2f> &corners)
{
    double distInPixels = norm(corners[0] - corners[patternSize.width]);
    double distInRealUnits = squareSize;
    return distInRealUnits / distInPixels;
}

// Function to compute the distance in real-world units between two points
double computeDistanceRealUnits(const Point2f &p1, const Point2f &p2, double scaleFactor)
{
    double distInPixels = norm(p1 - p2);
    double distInRealUnits = distInPixels * scaleFactor;
    return distInRealUnits;
}
