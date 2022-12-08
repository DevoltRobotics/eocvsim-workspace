import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class ConeTrackingPipeline extends OpenCvPipeline {

    public Scalar lowerRed = new Scalar(0, 0, 0);
    public Scalar upperRed = new Scalar(255, 255, 255);

    public Scalar lowerBlue = new Scalar(49, 96, 53);
    public Scalar upperBlue = new Scalar(119, 212, 255);

    private Mat redThreshold = new Mat();
    private Mat blueThreshold = new Mat();

    @Override
    public Mat processFrame(Mat input) {
        List<MatOfPoint> redContours = thresholdAndCountors(input, redThreshold, Imgproc.COLOR_RGB2HSV, lowerRed, upperRed);
        List<MatOfPoint> blueContours = thresholdAndCountors(input, blueThreshold, Imgproc.COLOR_RGB2HSV, lowerBlue, upperBlue);

        Imgproc.drawContours(input, redContours, -1, new Scalar(255, 0, 0), 3, 8);
        Imgproc.drawContours(input, blueContours, -1, new Scalar(0, 0, 255), 3, 8);

        Rect biggestBlueRect = null;

        for(MatOfPoint points : blueContours) {
            Rect rect = Imgproc.boundingRect(points);

            if(biggestBlueRect == null || rect.area() > biggestBlueRect.area()) {
                biggestBlueRect = rect;
            }
        }

        if(biggestBlueRect != null) {
            Imgproc.rectangle(input, biggestBlueRect, new Scalar(0, 0, 255));
        }

        return input;
    }

    private Mat cvtMat = new Mat();

    public List<MatOfPoint> thresholdAndCountors(Mat input, Mat thresholdOutput, int conversion, Scalar min, Scalar max) {
        Imgproc.cvtColor(input, cvtMat, conversion);
        Core.inRange(cvtMat, min, max, thresholdOutput);

        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(thresholdOutput, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        return contours;
    }
}
