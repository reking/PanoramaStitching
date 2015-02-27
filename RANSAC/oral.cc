#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<ctime>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/nonfree/nonfree.hpp>


using namespace std;
using namespace cv;
#define T_DIST 2

//Normalize Feature Points Coordinates before DLT Implementation
vector<Point2f> Normalize(const vector<Point2f> & Key, Mat &T)
{
    vector<Point2f> KeyNorm( Key );
    int Num = Key.size();
     if(Num > 2000)
         Num = 2000;
    int i = 0;
    double scale, tx, ty;//scale, translation x, y
    double xm,ym; //The mean value along x,y coordinate
    for(i = 0; i < Num; i++)
    {
        xm += Key[i].x;
        ym += Key[i].y;
    }
    cout<<"The xm sum is"<<xm<<endl;
    xm = static_cast<double>(xm / Num);
    ym = static_cast<double>(ym / Num);

  //  cout<<"The xm value is: "<<xm<<endl;
    double range = 0.0;
    for(i = 0; i < Num; i++)
    {
        range += std::sqrt(std::pow(Key[i].x - xm, 2.0) + std::pow(Key[i].y - ym, 2.0));
    }
    range = static_cast<double>(range / Num);
    scale = std::sqrt(2.0) / range;
    tx = -scale * xm;
    ty = -scale * ym;

    T = Mat::zeros(3,3,CV_64FC1);
    T.at<double>(0,0) = scale;
    T.at<double>(0,2) = tx;
    T.at<double>(1,1) = scale;
    T.at<double>(1,2) = ty;
    T.at<double>(2,2) = 1.0;
    //Note the operation is under inhomography coordinates, ommit the third
    //parameter which equals 1
    Mat X(3,1,CV_64FC1);
    Mat Y(3,1,CV_64FC1);
    X = Mat::zeros(3,1,CV_64FC1);
    Y = Mat::zeros(3,1,CV_64FC1);
    for(i = 0; i < Num; i++)
    {
        X.at<double>(0,0) = Key[i].x;
        X.at<double>(1,0) = Key[i].y;
        X.at<double>(2,0) = 1;
        Y = T*X;
        KeyNorm[i].x = Y.at<double>(0,0) / Y.at<double>(2,0);
        KeyNorm[i].y = Y.at<double>(1,0) / Y.at<double>(2,0);
    }
    cout<<"The range value is: "<<range<<endl;
    cout<<"Normalize is OK"<<endl;
    return KeyNorm;
}

//Feature Corresponding Points Calculated
void FeatureExtraction(const Mat& img1, const Mat& img2, vector<Point2f> & Key1, vector<Point2f> & Key2)
{
    Mat M1(img1);
    Mat M2(img2);
    vector<KeyPoint>  Keypoint1;
    vector<KeyPoint>  Keypoint2;
    SurfFeatureDetector detector(400);
    //detecting keypoints
    detector.detect(img1,Keypoint1);
    detector.detect(img2,Keypoint2);
    //computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptor1, descriptor2;
    extractor.compute(img1, Keypoint1, descriptor1);
    extractor.compute(img2, Keypoint2, descriptor2);

    cout<<"Keypoint1 x value is: "<<Keypoint1[10].pt.x<<endl;
    //matching descriptors
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    double max_dist = 0;
    double min_dist = 100;
    cout<<"The Feature points number is:"<<matches.size()<<endl;
    for (int i = 0; i < descriptor1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    cout<<"The Max dist and Min Dist are : "<< max_dist<<" "<<min_dist<<endl;
    for(int i = 0; i < descriptor1.rows; i++)
    {
        if(matches[i].distance < 3*min_dist && Key1.size() < 2000)
        {
            Key1.push_back(Keypoint1[ matches[i].queryIdx ].pt);
            Key2.push_back(Keypoint2[ matches[i].trainIdx ].pt);
        }
    }
    cout<<"The Key1 size and Key2 size are:"<<Key1.size()<<" "<<Key2.size()<<endl;
//    cout<<"The Key value"<<Key1[10].x<<endl;
    M1.release();
    M2.release();
    descriptor1.release();
    descriptor2.release();
    cout<<"Feature Extraction is OK"<<endl;
}
//Compute the Homography Matrix through DLT on 4 corresponding pairs of keypoints
void DLT(Mat& H, const vector<Point2f> & KeyNorm1, const vector<Point2f> & KeyNorm2, const vector<int> &orderin)
{
    int rowN = 2*orderin.size();
//    cout<<"orderin size is:"<<orderin[0]<<" "<<orderin[1]<<" "<<orderin[2]<< " "<< orderin[3]<<endl;
    Mat A= Mat::zeros(rowN,9,CV_64FC1);
    Mat U(rowN,rowN,CV_64FC1);
    Mat D(rowN,9,CV_64FC1);
    Mat VT(9,9,CV_64FC1);
    //Mat V(9,9,CV_64FC1);

    for(int i = 0; i < orderin.size(); i++)
    {
        A.at<double>(2*i, 3) = -KeyNorm1[ orderin[i] ].x;
        A.at<double>(2*i, 4) = -KeyNorm1[ orderin[i] ].y;
        A.at<double>(2*i, 5) = -1;
        A.at<double>(2*i, 6) = KeyNorm2[ orderin[i] ].y * KeyNorm1[ orderin[i] ].x;
        A.at<double>(2*i, 7) = KeyNorm2[ orderin[i] ].y * KeyNorm1[ orderin[i] ].y;
        A.at<double>(2*i, 8) = KeyNorm2[ orderin[i] ].y;

        A.at<double>(2*i+1, 0) =  KeyNorm1[ orderin[i] ].x;
        A.at<double>(2*i+1, 1) =  KeyNorm1[ orderin[i] ].y;
        A.at<double>(2*i+1, 2) = 1;
        A.at<double>(2*i+1, 6) = -KeyNorm2[ orderin[i] ].x * KeyNorm1[ orderin[i] ].x;
        A.at<double>(2*i+1, 7) = -KeyNorm2[ orderin[i] ].x * KeyNorm1[ orderin[i] ].y;
        A.at<double>(2*i+1, 8) = -KeyNorm2[ orderin[i] ].x;
        //cout<<"keynorm at 2i, 8 : "<<KeyNorm2[ orderin[i] ].x<<endl;;

    }
    cout<<"SVD is running now"<<endl;
    SVD::compute(A, D, U, VT,SVD::FULL_UV);//note the V is the V^T
    cout<<"SVD is OK"<<endl;
    //The homography matrix from the last column of V
    //cv::transpose(VT,V);
    for(int i = 0; i < 9; i++)
    {
        H.at<double>(i/3, i%3) = VT.at<double>(8,i);
//       cout<<"The H from DLt: "<<VT.at<double>(8,i)<<endl;
    }

       cout<<"The H from DLt: "<<VT.at<double>(8,8)<<endl;
    A.release();
    U.release();
    D.release();
    VT.release();
    cout<<"DLT is OK"<<endl;
}

int InliersCount(const Mat& H, const vector<Point2f> & KeyNorm1, const vector<Point2f> & KeyNorm2, vector<int> & inliers, double & dist_std)
{
    double Current_Dist, Sum_Dist, Mean_Dist;
    int numInlier = 0;
    int num = 0;
    int i;
    if(KeyNorm1.size() == KeyNorm2.size())
         num = KeyNorm1.size();
    
    cout<<"KeyNorm2 is: "<<KeyNorm2[11].x<<endl;
    Mat x(3,1,CV_64FC1);
    Mat xp(3,1,CV_64FC1);
    Mat Mid(3,1,CV_64FC1);
    Mat Dist(num,1,CV_64FC1);
    Point2f  Temp;
    Mat invH(3,3,CV_64FC1);
    cv::invert(H, invH);
    cout << "The for loop number: "<< num<<endl;
    for(i = 0; i < num; i++)
    {
        Current_Dist = 0.0;
       x.at<double>(0,0) = KeyNorm1[i].x;
       x.at<double>(1,0) = KeyNorm1[i].y;
       x.at<double>(2,0) = 1;

       xp.at<double>(0,0) = KeyNorm2[i].x;
       xp.at<double>(1,0) = KeyNorm2[i].y;
       xp.at<double>(2,0) = 1;
       
      Mid = H*x;
    //  cout<<"The mid value is: "<<Mid.at<double>(0,2)<<endl;
      Temp.x = static_cast<int>(Mid.at<double>(0,0) / Mid.at<double>(2,0));
      Temp.y = static_cast<int>(Mid.at<double>(1,0) / Mid.at<double>(2,0));
      //cout<<"Temp.x is :"<<Temp.x<<endl;
      //cout<<"KeyNorm2 is: "<<KeyNorm2[i].x<<endl;
      Current_Dist += std::pow(Temp.x - KeyNorm2[i].x, 2.0) + std::pow(Temp.y - KeyNorm2[i].y, 2.0);
      Mid = invH*xp;
      Temp.x = static_cast<int>(Mid.at<double>(0,0) / Mid.at<double>(2,0));
      Temp.y = static_cast<int>(Mid.at<double>(1,0) / Mid.at<double>(2,0));

      Current_Dist += std::pow(Temp.x - KeyNorm1[i].x, 2.0) + std::pow(Temp.y - KeyNorm1[i].y, 2.0);
    //cout<<"The current dist is: "<<Current_Dist<<endl;
     if(Current_Dist < T_DIST)
     {
         numInlier++;
         inliers.push_back(i);
         Dist.at<double>(i,0) = Current_Dist;
         Sum_Dist += Current_Dist;
     }
    }

    dist_std = 0;
    Mean_Dist = Sum_Dist / (double) numInlier;
    int j =0;
    for(i = 0; i < inliers.size(); i++)
    {
        j = inliers[i];
        dist_std += std::pow(Dist.at<double>(j,0) - Mean_Dist, 2.0);
    }
    dist_std /= static_cast<double>(numInlier - 1);
    x.release();
    xp.release();
    Mid.release();
    Dist.release();
    invH.release();

    cout<<"Inlier Count is OK, inlier number is: "<<numInlier<<endl;
    return numInlier;
}

bool isColinear(const vector<int> & orderin, const vector<Point2f> & Key1)
{
    bool iscolinear;
    int num = orderin.size();
    double value;
       Mat p1(3,1,CV_64FC1);
       Mat p2(3,1,CV_64FC1);
       Mat p3(3,1,CV_64FC1);
       Mat lin(3,1,CV_64FC1);
    iscolinear = false;

    for( int i = 0; i < num-2; i++)
    {
        p1.at<double>(0,0) = Key1[orderin[i]].x;
        p1.at<double>(1,0) = Key1[orderin[i]].y;
        p1.at<double>(2,0) = 1;
        for( int j = i+1; j < num -1; j++)
        {
           p2.at<double>(0,0) = Key1[orderin[j]].x;
           p2.at<double>(1,0) = Key1[orderin[j]].y;
           p2.at<double>(2,0) = 1;
           lin = p1.cross(p2);
           for(int k = j+1; k < num; k++)
           {
               p3.at<double>(0,0) = Key1[orderin[k]].x;
               p3.at<double>(1,0) = Key1[orderin[k]].y;
               p3.at<double>(2,0) = 1;
               value = p3.dot(lin);
               if(std::abs(value) < 10e-2)
               {
                   iscolinear = true;
                   break;
               }
           }
           if(iscolinear == true) break;
        }
    }
    p1.release();
    p2.release();
    p3.release();
    lin.release();
    return iscolinear;
}


void RANSAC_DLT(const vector<Point2f> & Key1, const vector<Point2f> & Key2, Mat & H_final)
{
    int numAll = 0;
    if(Key1.size() == Key2.size())
        numAll = Key1.size();
    cout<<"The number of keypoints: "<<numAll<<endl;
    int N = 500, nu = 4;// The whole sample points and the 4 points to estimate
    int Max_num = 4; //Max number of inliers 
    int sample = 0;
    int numInlier;
    double e, p = 0.99;
    double current_dist_std,dist_std;
    vector<int> orderin(nu,0);
    vector<int> inliers;
    vector<int> max_inliers;
    //H_final = Mat(3,3,CV_64FC1);
    Mat T1(3,3,CV_64FC1);
    Mat T2(3,3,CV_64FC1);
    Mat InvT2(3,3,CV_64FC1);
    Mat H(3,3,CV_64FC1);
    srand(unsigned(time(0)));
    bool iscolinear = true;
    vector<Point2f> KeyNorm1 =  Normalize(Key1, T1);
    vector<Point2f> KeyNorm2 =  Normalize(Key2, T2);


    while(N > sample)
    {
        iscolinear = true;
        while(iscolinear == true)
        {
          iscolinear = false;
          for(int i = 0; i < nu; i++)
          {
            orderin[i] = rand() % numAll;
          }
          if(iscolinear == false)
              iscolinear = isColinear(orderin, Key1);
        }

        DLT(H,KeyNorm1,KeyNorm2, orderin);

        numInlier = InliersCount(H,KeyNorm1,KeyNorm2, inliers, dist_std);

        if(numInlier > Max_num || (numInlier == Max_num &&  current_dist_std > dist_std))
        {
            Max_num = numInlier;
            max_inliers.clear();
            //max_inliers.swap(vector<int>());
            max_inliers.resize(0);
            if(max_inliers.size() == 0)
                cout<<"Whether max_inliers is empty: "<<max_inliers.size()<<endl;
            max_inliers.swap(inliers);
            H_final = H;
            //max_inliers = inliers;
            current_dist_std = dist_std;
           
        }
        //update the parameters for N
        e = 1 - (double)numInlier / (double) numAll;
        //cout<<"The numInlier is: "<<numInlier<<endl;
        //cout<<"The numall is: " << numAll <<endl;
        N = static_cast<int>( std::log(1-p) / std::log(1-std::pow(1-e,nu)));
        cout<<"The calculated N is : "<< N <<endl;
            inliers.clear();
            inliers.resize(0);
            if(inliers.size() == 0)
                cout<<"Whether inliers is empty:" <<inliers.size()<<endl;

        sample++;
        cout<<"sample: "<< sample<<endl;
    }
    cv::invert(T2,InvT2);
    // H = InvT2* H;
    //H = H*T1;


    cout<<"The N times selection is OK Now"<<endl;
    cout<<"The max_inlier number is: "<< max_inliers.size() <<endl;
    DLT(H_final,KeyNorm1,  KeyNorm2, max_inliers);
     for(int i = 0; i < 9; i++)
    {
        cout<<"The invert T2 value is "<<InvT2.at<double>(i/3,i%3)<<endl;
    }
    for(int i = 0; i < 9; i++)
    {
        cout<<"The  T1 value is "<<T1.at<double>(i/3,i%3)<<endl;
    }

    H_final = InvT2 * H_final;
    H_final = H_final * T1;
    cout<<"Ransac is OK"<<endl;
}
void  LinearBlendPrepare( Mat& img1,  Mat& img2)
{
    int row = img1.rows;
    int col = img1.cols;
    if(img1.size() == img2.size())
        cout<<"The Image size equals"<<endl;
    Mat bingo(3,1,CV_64FC1);
    Mat temp(3,1,CV_64FC1);
    Vec3b intensity;
    Vec3b liability;
    double blue,green,red;
    double blue1,green1,red1;
    for(int i = 0; i< row-1; i++)
        for(int j = 0; j< col-1; j++)
        {
            liability = img1.at<Vec3b>(i,j);
            intensity = img2.at<Vec3b>(i,j);
            blue = intensity.val[0];
            green = intensity.val[1];
            red =  intensity.val[2];

            blue1 = liability.val[0];
            green1 = liability.val[1];
            red1 = liability.val[2];

            if( (blue+green+red) != 0.0 && (blue1 + green1 + red1) != 0.0)
            {
                img1.at<Vec3b>(i,j) /= 2.0;
                img2.at<Vec3b>(i,j) /= 2.0;
            }
        }
    
                    cout<<"At Least OK now"<<endl;
}

int main(int argc, char *argv[])
{
    if(argc < 5)
        cout<<"At least four Image files:"<<endl;

    Mat Image0, Image1, Image2, Image3, Image4;
    Image1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Image2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    Image3 = cv::imread(argv[3], CV_LOAD_IMAGE_COLOR);
    Image4 = cv::imread(argv[4], CV_LOAD_IMAGE_COLOR);
    Image0 = Image1;
    //string ty = type2str(Image1.type());
    //cout<<"The image type is: "<<ty<<endl;
    cout<<"The image depth is: "<<Image1.channels()<<endl;
   // Mat Trans;

    Mat img1, img2,img3,img4;
    cv::resize(Image1,Image1,Size(800,600));
    cv::resize(Image2,Image2,Size(800,600));
    cv::resize(Image3,Image3,Size(800,600));
    cv::resize(Image4,Image4,Size(800,600));
    cv::cvtColor(Image1,img1,CV_BGR2GRAY);
    cv::cvtColor(Image2,img2,CV_BGR2GRAY);
    cv::cvtColor(Image3,img3,CV_BGR2GRAY);
    cv::cvtColor(Image4,img4,CV_BGR2GRAY);
    //feature extracted to two vectors
    vector<Point2f> Key1;
    vector<Point2f> Key2;
    vector<Point2f> Key21;
    vector<Point2f> Key22;
    vector<Point2f> Key3;
    vector<Point2f> Key4;

    FeatureExtraction(img1, img2, Key1, Key2);
    FeatureExtraction(img3, img2, Key3, Key21);
    FeatureExtraction(img4, img3, Key4, Key22);

    Mat H_final(3,3,CV_64FC1);
    Mat H_final2(3,3,CV_64FC1);
    Mat H_final3(3,3,CV_64FC1);
    Mat H_trans(3,3,CV_64FC1);


    H_trans = Mat::eye(3,3,CV_64FC1);
    H_trans.at<double>(0,2) = 200.0;
    H_trans.at<double>(0,0) = 0.6;
    H_trans.at<double>(1,1) = 0.6;

    RANSAC_DLT(Key1, Key2, H_final);
    RANSAC_DLT(Key3, Key21, H_final2);
    RANSAC_DLT(Key4, Key22, H_final3);


    Mat H_show = H_trans*H_final;
    Mat H_show2 = H_trans*H_final2;
    Mat H_show3 = H_trans*H_final2*H_final3;

    for(int i = 0; i < 9; i++)
    {
        cout<<"The H value is "<<H_final.at<double>(i/3,i%3)<<endl;
    }
    cout<<"The Image size is: "<<Image1.size()<<endl;

    Mat Trans;//(2*Image1.rows,2*Image1.cols,CV_8UC3);
    Mat Trans2;
    Mat Trans3;
    Mat opera1, opera2, opera3, opera4;
    Image1.copyTo(opera1);
    Image2.copyTo(opera2);
    Image3.copyTo(opera3);
    Image4.copyTo(opera4);

    cv::warpPerspective(opera1,Trans,H_show,Image0.size(),INTER_LINEAR,BORDER_CONSTANT);
    cv::warpPerspective(opera2,Image2,H_trans,Image0.size(),INTER_LINEAR,BORDER_CONSTANT);
    cv::warpPerspective(opera3,Trans2,H_show2,Image0.size(),INTER_LINEAR,BORDER_CONSTANT);
    cv::warpPerspective(opera4,Trans3,H_show3,Image0.size(),INTER_LINEAR,BORDER_CONSTANT);



    double beta = 1.0;
    double alpha = 1.0;
    Mat finalImage(2400,1200,CV_8UC3);

    LinearBlendPrepare(Trans, Image2);
    addWeighted(Image2,alpha,Trans,beta,0.0,finalImage);
    cout<<"Blend 1 is OK"<<endl;

    LinearBlendPrepare(finalImage,Trans2);
    addWeighted(finalImage,alpha,Trans2,beta,0.0,finalImage);
    cout<<"Blend 2 is OK"<<endl;

    LinearBlendPrepare(finalImage,Trans3);
    addWeighted(finalImage,alpha,Trans3,beta,0.0,finalImage);
    cout<<"Blend 3 is OK"<<endl;


    imwrite("finalImage.jpg",finalImage);
    cout<<"The Trans size is:  "<<Trans.size()<<endl;
    namedWindow("Linear Blend",WINDOW_AUTOSIZE);
    namedWindow("Left Image",WINDOW_AUTOSIZE);
    namedWindow("Right Image",WINDOW_AUTOSIZE);
    imshow("Left Image",Trans);
    imshow("Right Image",Image2);

    imshow("Linear Blend", finalImage);
    
    waitKey(0);
    return 0;
}
