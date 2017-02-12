/* People counter from a CCTV camera
 * Applicable in low density scene
 * Copyright (C) 2017  Author Ajay Soni(ajaysonidip@gmail.com)
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/background_segm.hpp>
#include <cstdio>
#include <iostream>
using namespace std;
using namespace cv;

static void help(char** argv)
{
    cout << "\nDemonstrate People counting from a CCTV footage.\n"
         << "Call:\n   " << argv[0] << " Video\n"
         << "People counter from a CCTV camera\n"
         << "Applicable in low density scene\n"
         << "Copyright (C) 2017  Author Ajay Soni(ajaysonidip@gmail.com)"
         << endl;
}

bool isRedundantIdx(vector<int> vecRedundant, int idx1, int idx2)
{

    for(int vecItr = 0; vecItr < vecRedundant.size(); vecItr++)
    {
        if(vecRedundant.at(vecItr) == idx1 || vecRedundant.at(vecItr) == idx2)
        {
            return true;
        }
    }
    return false;

}

int main(int argc, char *argv[])
{

    if( argc !=2 )
    {
        help(argv);
        return -1;
    }

    //Declare video capture object
    VideoCapture capture;
    //Open a video stream
    //    capture.open(0);
    capture.open(argv[1]);
    if(!capture.isOpened())
    {
        return -1;
    }
    //Declare a matrix for taking the frame from video capture
    Mat input_image, smaller_input;
    //Declare a matrix for converting rgb to gray image
    Mat  pMog_mask,pMog_mask_y,pMog_mask_red, pMog_mask_green, pMog_mask_blue, morph_img;

    Ptr<BackgroundSubtractorMOG> pMog_red = new BackgroundSubtractorMOG();
    Ptr<BackgroundSubtractorMOG> pMog_blue = new BackgroundSubtractorMOG();
    Ptr<BackgroundSubtractorMOG> pMog_green = new BackgroundSubtractorMOG();
    Ptr<BackgroundSubtractorMOG> pMog_input = new BackgroundSubtractorMOG();


    cv::Mat struct_elem = cv::Mat(7,7,CV_8UC1,1);
    cv::Mat struct_elem2 = cv::Mat(3,3,CV_8UC1,1);
    vector<Mat> channels;
    char key;
    capture >> input_image;

    vector<Point> roi_crossing_blobs_prev;

    int left_crossing_counter = 0, right_crossing_counter = 0;

    while(1)
    {
        //copy video stream in a mat
        capture >> input_image;
        //check if mat is not empty
        if(!input_image.empty())
        {

            // ALgo Step 1- Subtract background to get some blobs

            //convert input(BGR) to gray
            Rect smallerRoi = Rect(30,0,230,input_image.rows); /**< TODO */
            smaller_input = Mat(input_image,smallerRoi);
            //input_image = smaller_input;
            split(input_image,channels);

            pMog_blue->operator()(channels[0], pMog_mask_blue);
            pMog_green->operator()(channels[1], pMog_mask_green);
            pMog_red->operator()(channels[2], pMog_mask_red);
            pMog_input->operator ()(input_image,pMog_mask_y);


            pMog_mask = pMog_mask_red+pMog_mask_blue+pMog_mask_green;

            // Step 2- Morphological operations to remove noises


            cv::morphologyEx(pMog_mask,morph_img,CV_MOP_DILATE,struct_elem);
            cv::morphologyEx(morph_img,morph_img,CV_MOP_OPEN,struct_elem2);
            cv::morphologyEx(morph_img,morph_img,CV_MOP_ERODE,struct_elem);


            // Step 3-Find countours remove noises by area, height/width ratio

            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;

            findContours( morph_img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            double areraThresh = 2.0;
            double rect_height_width_ratio = 0.05;
            vector<Rect> bounding_rects, bounding_rects_filt, bounding_rects_filt2;
            Mat filtered_img = Mat::zeros(input_image.rows,input_image.cols, CV_8UC1);

            for(int ctrItr=0; ctrItr < contours.size() ; ctrItr++)
            {
                if(contourArea(contours.at(ctrItr)) > areraThresh)
                {
                    Rect temp_rect = boundingRect(contours.at(ctrItr));
                    if(temp_rect.height/temp_rect.width > rect_height_width_ratio)
                    {
                        bounding_rects.push_back(temp_rect);
                    }
                }
            }

            // Step -4 Check if there are rects inside rects if yes merge them
            // To check if rects are inside other bounding rect and
            //if they are in the near vicinity of other rects

            vector<int> index_redundant_rects;
            index_redundant_rects.push_back(-1);

            int equDist_thresh = 58;

            if(bounding_rects.size() > 0)
            {
                for(int rectItr=0; rectItr < bounding_rects.size() ; rectItr++)
                {
                    for(int rectItr2=0; rectItr2 < bounding_rects.size() ; rectItr2++)
                    {
                        bool isRed = isRedundantIdx(index_redundant_rects,rectItr,rectItr2);

                        if(rectItr != rectItr2 && !isRed)
                        {
                            Rect first_rect, second_rect;
                            first_rect = bounding_rects.at(rectItr);
                            second_rect= bounding_rects.at(rectItr2);

                            Point first_rect_center = Point((first_rect.x+first_rect.width/2),
                                                            (first_rect.y+first_rect.height/2));
                            Point second_rect_center = Point((second_rect.x+second_rect.width/2),
                                                             (second_rect.y+second_rect.height/2));
                            if(first_rect.contains(second_rect_center))
                            {
                                index_redundant_rects.push_back(rectItr2);
                            }
                            else if(second_rect.contains(first_rect_center))
                            {
                                index_redundant_rects.push_back(rectItr);
                            }
                        }
                    }
                }
            }

            for(int rectItr2=0; rectItr2 < bounding_rects.size() ; rectItr2++)
            {
                if(!isRedundantIdx(index_redundant_rects,rectItr2, -2))
                {
                    bounding_rects_filt.push_back(bounding_rects.at(rectItr2));
                }
            }

            vector<int> mergedIndices;

            for(int rectItr=0; rectItr < bounding_rects_filt.size() ; rectItr++)
            {
                vector<int> toBeMergedIdices;

                if(!(isRedundantIdx(mergedIndices,rectItr,-2)))
                {

                    for(int rectItr2=0; rectItr2 < bounding_rects_filt.size() ; rectItr2++)
                    {
                        if(!(isRedundantIdx(mergedIndices,rectItr2,-2)))
                        {
                            Rect first_rect, second_rect;
                            first_rect = bounding_rects_filt.at(rectItr);
                            second_rect= bounding_rects_filt.at(rectItr2);

                            Point first_rect_center = Point((first_rect.x+first_rect.width/2),
                                                            (first_rect.y+first_rect.height/2));
                            Point second_rect_center = Point((second_rect.x+second_rect.width/2),
                                                             (second_rect.y+second_rect.height/2));


                            int euqDist = sqrt(pow((first_rect_center.x-second_rect_center.x),2)+pow((first_rect_center.y-second_rect_center.y),2));


                            if(euqDist < equDist_thresh)
                            {
                                toBeMergedIdices.push_back(rectItr2);
                                mergedIndices.push_back(rectItr2);
                            }

                        }
                    }
                }


                if(toBeMergedIdices.size() > 0)
                {
                    Point topLeft = Point(input_image.cols, input_image.rows);
                    Point bottomRight = Point(0,0);
                    for(int mergInd = 0; mergInd < toBeMergedIdices.size(); mergInd++)
                    {
                        Rect tmpRect = bounding_rects_filt.at(toBeMergedIdices.at(mergInd));


                        topLeft.x = tmpRect.tl().x < topLeft.x ? tmpRect.tl().x : topLeft.x;
                        topLeft.y = tmpRect.tl().y < topLeft.y ? tmpRect.tl().y : topLeft.y;
                        bottomRight.x = tmpRect.br().x > bottomRight.x ? tmpRect.br().x : bottomRight.x;
                        bottomRight.y = tmpRect.br().y > bottomRight.y ? tmpRect.br().y : bottomRight.y;
                    }


                    Rect mergedRect = Rect(topLeft.x,topLeft.y, bottomRight.x-topLeft.x, bottomRight.y-topLeft.y);
                    rectangle(input_image,mergedRect,Scalar(0,0,255),2);

                    bounding_rects_filt2.push_back(mergedRect);

                }
            }


            // Step 6 - Logic for blobs crossing an roi say a door
            // Log Centers in roi region and see which side they are moving
            // and increment the respective counter

            int centerOffset = 5;
            int roiWidth = 20;

            Rect roiRect = Rect(((input_image.cols/2)-centerOffset),
                                0,roiWidth,input_image.rows);

            int dist_from_prev_pt_thresh = 7;

            vector<int> prev_idx_tobe_erased;
            vector<Point> current_centers_tobe_pushed;

            for(int recItr=0; recItr < bounding_rects_filt2.size(); recItr++)
            {
                Rect filtRect = bounding_rects_filt2.at(recItr);
                Point currentCenter =   Point((filtRect.x+filtRect.width/2),
                                              (filtRect.y+filtRect.height/2));

                if(roiRect.contains(currentCenter))
                {
                    int stop =0;
                }

                double match_found = false;
                if(roi_crossing_blobs_prev.size() > 0)
                {
                    for(int prevCenItr=0; prevCenItr< roi_crossing_blobs_prev.size(); prevCenItr++)
                    {

                        Point prevCent = roi_crossing_blobs_prev.at(prevCenItr);

                        int currentToPrevDist = sqrt(pow((currentCenter.x-prevCent.x),2)+pow((currentCenter.y-prevCent.y),2));

                        if(currentToPrevDist < dist_from_prev_pt_thresh)
                        {

                            match_found = true;
                            if(roiRect.contains(currentCenter))
                            {
                                roi_crossing_blobs_prev.at(prevCenItr) = currentCenter;
                            }
                            else if(currentCenter.x > (roiRect.x+roiRect.width))
                            {

                                prev_idx_tobe_erased.push_back(prevCenItr);
                                right_crossing_counter++;

                            }
                            else if(currentCenter.x < roiRect.x)
                            {
                                prev_idx_tobe_erased.push_back(prevCenItr);
                                left_crossing_counter++;

                            }
                        }

                    }
                }

                if(!match_found && roiRect.contains(currentCenter))
                {
                    current_centers_tobe_pushed.push_back(currentCenter);
                }

            }

            vector<Point> updated_prev_centers;
            for(int prevIdxItr= 0 ; prevIdxItr < roi_crossing_blobs_prev.size(); prevIdxItr++)
            {

                bool found_idx = false;
                for(int erasedItr=0; erasedItr< prev_idx_tobe_erased.size(); erasedItr++)
                {

                    if(prev_idx_tobe_erased.at(erasedItr) == prevIdxItr)
                        found_idx = true;

                }
                if(!found_idx)
                {
                    updated_prev_centers.push_back(roi_crossing_blobs_prev.at(prevIdxItr));
                }
            }

            roi_crossing_blobs_prev.clear();
            updated_prev_centers.insert(updated_prev_centers.end(),
                                        current_centers_tobe_pushed.begin(),
                                        current_centers_tobe_pushed.end());

            roi_crossing_blobs_prev = updated_prev_centers;

            rectangle(input_image,roiRect,Scalar(0,255,0),2);



            char leftCrossing[20], rightCrossing[20];

            sprintf(leftCrossing,"%d",left_crossing_counter);
            sprintf(rightCrossing, "%d", right_crossing_counter);

            putText(input_image,leftCrossing,Point(20,50),CV_FONT_HERSHEY_COMPLEX,1.0,Scalar(255,255,255),2);
            putText(input_image,rightCrossing,Point(input_image.cols-20,50),CV_FONT_HERSHEY_COMPLEX,1.0,Scalar(255,255,255),2);

            imshow("People Counter",input_image);

            key = waitKey(1);
            if(key == 'q' || key == 'Q')
                break;
            if(key == 'p')
                key = waitKey(0);
            if(key == 'q' || key == 'Q')
                break;
        }
        else
        {
            break;
        }
    }
}
