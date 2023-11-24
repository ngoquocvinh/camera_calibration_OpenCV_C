# camera_calibration_OpenCV_C
 This program captures video from a camera, detects a chessboard pattern, calibrates the camera,
  undistorts an image, and then processes a video stream to find and measure distances between objects.

  How to use
  1. prepare chessboard with 10x7 squares. each square is 22x22mm
  2. install Droidcam on iphone and run it to get the ip like: http://192.168.1.6:4747
  3. the iphone camera and computer must be a same network
  4. fixed camera positon and use chessboard for calibrication
  5. put the chessboard in the camera view and press spacebar key to capture a sample.
  6. continue do it 12 times to gather 12 sample for the internal and external calibrication matrix calculation
  7. press the spacebar for calculate two farthest distance, compare with your real manually measure in milimeter
  8. press the spacebar and put two blue object to detect distance between them. Rotate one of them for visualling
  9. the distance may not correct, please check your calibration methods
