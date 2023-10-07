#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <iostream>

using namespace cv;
using namespace xfeatures2d;
using namespace std;
using namespace cv::cuda;

#include <iostream>
#include <omp.h>

int serial(int npic){
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    double t1 = omp_get_wtime( );
    for(int i = 0; i < npic; i++) {
        vector<KeyPoint> keypoints;
        Mat im;
        char filename[32] = {0};
        sprintf(filename, "test%d.jpg", i+1);
        im = imread(filename, IMREAD_GRAYSCALE);
        detector->detect(im, keypoints, Mat());
        std::cout << "find " << keypoints.size() << " keypoints in " << filename << std::endl;
        Mat im_matches;
        drawKeypoints(im, keypoints, im_matches, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        sprintf(filename, "test%d_out.jpg", i+1);
        imwrite(filename, im_matches);
    }
    double t2 = omp_get_wtime( );
    std::cout<<"time: "<<t2-t1<<std::endl;
    return 0;
}

int openmp(int npic){
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    double t1 = omp_get_wtime( );
    //#pragma omp parallel for num_threads(npic)
    #pragma omp parallel for
    for(int i = 0; i < npic; i++) {
        printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
        vector<KeyPoint> keypoints;
        Mat im;
        char filename[32] = {0};
        sprintf(filename, "test%d.jpg", i+1);
        im = imread(filename, IMREAD_GRAYSCALE);
        detector->detect(im, keypoints, Mat());
        std::cout << "find " << keypoints.size() << " keypoints in " << filename << std::endl;
        Mat im_matches;
        drawKeypoints(im, keypoints, im_matches, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        sprintf(filename, "test%d_out_openmp.jpg", i+1);
        imwrite(filename, im_matches);
    }
    double t2 = omp_get_wtime( );
    std::cout<<"time: "<<t2-t1<<std::endl;
    return 0;
}

int openmp_cuda(int npic){
    int minHessian = 400;
    printShortCudaDeviceInfo(getDevice());
    double t1 = omp_get_wtime( );
    //#pragma omp parallel for num_threads(npic)
    #pragma omp parallel for
    for(int i = 0; i < npic; i++) {
        SURF_CUDA surf;
        printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
        Mat im;
        char filename[32] = {0};
        sprintf(filename, "test%d.jpg", i+1);
        im = imread(filename, IMREAD_GRAYSCALE);
        CV_Assert(!im.empty());
        GpuMat imGPU;
        imGPU.upload(im);
        CV_Assert(!imGPU.empty());
        GpuMat keypointsGPU;
        GpuMat descriptorsGPU;
        surf(imGPU, GpuMat(), keypointsGPU, descriptorsGPU);
        vector<KeyPoint> keypoints;
        surf.downloadKeypoints(keypointsGPU, keypoints);
        //std::cout << "find " << keypointsGPU.cols << " keypoints in " << filename << std::endl;
        std::cout << "find " << keypoints.size() << " keypoints in " << filename << std::endl;
        Mat im_matches;
        drawKeypoints(im, keypoints, im_matches, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        sprintf(filename, "test%d_out_openmp_cuda.jpg", i+1);
        imwrite(filename, im_matches);
    }
    double t2 = omp_get_wtime( );
    std::cout<<"time: "<<t2-t1<<std::endl;
    return 0;
}

int main() {
    int npic = 6;
    serial(npic);
    openmp(npic);
    openmp_cuda(npic);
    return 0;
}
/*
                                                    serial                          openmp                          openmp_cuda
6pics                                               2.21391                         2.01917                         0.642105
12pics                                              4.30375                         2.81234                         1.56068
24pics(12threads maybe by omp_get_num_procs)        9.50376                         6.3159                          3.26673
24pics(24threads)                                   9.30529                         6.26617                         3.32203
*/