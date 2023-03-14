#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
using namespace std;
map<string,double[4]>K;
map<string,double[6]>cam01_pose;
map<string,double[6]>Cam1ToCam_i;
map<string,int>cam01pose_timestamp2pose_index;
map<string,int>map_cam_num;
string str;
// Read a Bundle Adjustment in the Large dataset.
double mkpts[2][1000000];
double Points[3][1000000];
int cameras_index[1000000];
//int points_index[10000];
struct e{
    string timestamp;
    string cam;
}camera_index2[2000000];
int ii=0;
int Num_relative_cam, Num_timestamp, Num_points, Num_observations;
string video_num, path;
string root_path = "../../../save/optimized_pose/nuScenes/E/2frame_USAC_ACCURATE/";

void readFileJson()
{
    path = root_path + video_num;
    map_cam_num["L"]=0;
    map_cam_num["R"]=1;

    ifstream myfile_num(path + "/num.txt");
    myfile_num>>Num_relative_cam>>Num_timestamp>>Num_points>>Num_observations;
    myfile_num.close();

    ifstream myfile_cam01_pose(path + "/cam01_pose.txt");
    for(int i=0;i<Num_timestamp;i++)
    {
        myfile_cam01_pose>>str;
        for(int k=0;k<6;k++)
            myfile_cam01_pose>>cam01_pose[str][k];
    }
    myfile_cam01_pose.close();
    ifstream myfile_intrinsics(path + "/intrinsics.txt");
    for(int i=0;i<Num_relative_cam;i++) //6
    {
        myfile_intrinsics>>str;
        for(int k=0;k<4;k++)
            myfile_intrinsics>>K[str][k];
    }
    myfile_intrinsics.close();

    ifstream myfile_Cam1ToCam_i(path + "/Cam_1ToCam_i.txt");
    for(int i=0;i<Num_relative_cam;i++) //6
    {
        myfile_Cam1ToCam_i>>str;
        for(int k=0;k<6;k++){
            myfile_Cam1ToCam_i>>Cam1ToCam_i[str][k];
        }
    }
    myfile_Cam1ToCam_i.close();
    ifstream myfile_cameras_index(path + "/cameras_index.txt");
    for(int i=0;i<Num_observations;i++){
        myfile_cameras_index>>camera_index2[i].timestamp>>camera_index2[i].cam;
    }

    ifstream myfile_cameras_mkpts(path + "/mkpts.txt");
    for(int i=0;i<Num_observations;i++){
        myfile_cameras_mkpts>>mkpts[0][i]>>mkpts[1][i];
    }

    ifstream myfile_Points(path + "/mkpts.txt");
    for(int i=0;i<Num_observations;i++){
        myfile_Points>>Points[0][i]>>Points[1][i]>>Points[2][i];
    }
    myfile_Points.close();
    ifstream myfile_points(path + "/points.txt");
    for(int i=0;i<Num_points;i++)
        for(int k=0;k<3;k++)
        myfile_points>>Points[k][i];
    myfile_points.close();
    cout<<"read file finish"<<endl;
}

class BALProblem {
    public:

        int num_observations() const {return num_observations_;}
        const double* observations() const {return observations_;}
        double* mutable_cameras() {return parameters_;}

        //double* mutable_camera_for_observation(int i) {return parameters_ + camera_index_[i] * 6;}
        double* mutable_point_for_observation(int i) {
            return parameters_  + 6 * (num_relative_cam_+ num_timestamp_) + i * 3;}
        double* mutable_cam01_pose(string timestamp){
            return parameters_ + (num_relative_cam_+ cam01pose_timestamp2pose_index[timestamp])*6;}
        double* mutable_Pose_Cam01ToCam_i(string cam_name){
            return parameters_ + map_cam_num[cam_name] * 6;
        }

        bool LoadFile() {
            //num_cameras_ = Num_cameras;
            num_relative_cam_ = Num_relative_cam;
            num_timestamp_ = Num_timestamp;
            num_points_ = Num_points;
            num_observations_ = Num_observations;

            observations_ = new double[2 * num_observations_];

            num_parameters_ = 6 * (num_relative_cam_+num_timestamp_) + 3 * num_points_;  //num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
            parameters_ = new double[num_parameters_];

            for(int i = 0; i < num_observations_; ++i) {
                observations_[i*2+0] = mkpts[0][i];
                observations_[i*2+1] = mkpts[1][i];
            }
            ii=0;
            for(auto iter = Cam1ToCam_i.begin();iter!=Cam1ToCam_i.end();iter++){
                for(int k=0;k<6;k++){
                    parameters_[ii*6+k] = iter->second[k];
                    //cout<<ii*6+k<<"="<<iter->second[k]<<"\t"<<parameters_[ii*6+k]<<endl;
                }
                ii++;
            }
            ii=num_relative_cam_;
            for(auto iter = cam01_pose.begin();iter!=cam01_pose.end();iter++){
                cam01pose_timestamp2pose_index[iter->first] = ii-num_relative_cam_;
                for(int k=0;k<6;k++){
                    parameters_[ii*6+k] = iter->second[k];
                }
                ii++;
            }

            for(int i=0; i<num_points_;i++){
                int index = (num_relative_cam_ + num_timestamp_)*6 + i*3;
                parameters_[index+0] =  Points[0][i];
                parameters_[index+1] =  Points[1][i];
                parameters_[index+2] =  Points[2][i];
            }
            //debug
            //for(int i=0;i<100;i++)

            return true;
        }

    private:
        template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
            int num_scanned = fscanf(fptr, format, value);
            if (num_scanned != 1) {
                LOG(FATAL) << "Invalid UW data file.";
            }
        }
        int num_relative_cam_;  // Num_relative_cam, Num_timestamp,
        int num_timestamp_;
        //int num_cameras_;

        int num_points_;
        int num_observations_;
        int num_parameters_;

        //int* point_index_;
        int* camera_index_;
        double* observations_;
        double* parameters_;
};

/****************************************************************************************
 * 小孔相机模型.
 * camera用9个参数表示， 3 个表示旋转, 3 个表示平移, 1个焦距，2个径向畸变
 * 中心点没有建模，假设为图片中心
****************************************************************************************/

struct SnavelyReprojectionError_Cam_01 {
    SnavelyReprojectionError_Cam_01(double observed_x, double observed_y, string cam_name):observed_x(observed_x), observed_y(observed_y), cam_name(cam_name) {}
    template <typename T>
    bool operator()(const T* const point, T* residuals) const {
        //T p[3];
        //cout<<"points="<<point[0]<<" "<<point[1]<<" "<<point[2]<<endl;
        //for(int i=0;i<6;i++)
        //    cout<<"pose_"<<i<<"="<<cam01_pose[i]<<endl;
        //cout<<endl;
        //ceres::AngleAxisRotatePoint(cam01_pose, point, p);

        //cout<<"p after rotation="<<endl<<p[0]<<" "<<p[1]<<" "<<p[2]<<endl;

        //p[0] += cam01_pose[3];
        //p[1] += cam01_pose[4];
        //p[2] += cam01_pose[5];

        //cout<<"p after traslation="<<endl<<p[0]<<" "<<p[1]<<" "<<p[2]<<endl;

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = point[0] / point[2];
        T yp = point[1] / point[2];

        T predicted_x = K[cam_name][0]*xp + K[cam_name][2]; //focal * distortion * xp;    ##bug!!!
        T predicted_y = K[cam_name][1]*yp + K[cam_name][3]; //focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        //cout<<"x="<<predicted_x<<","<<observed_x<<","<<residuals[0]<<endl;
        //cout<<endl<<endl;
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y, const string cam_name) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Cam_01, 2, 3>(
                new SnavelyReprojectionError_Cam_01(observed_x, observed_y, cam_name)));
  }
    double observed_x;
    double observed_y;
    string cam_name;

};


struct SnavelyReprojectionError_Cam_else {
    SnavelyReprojectionError_Cam_else(double observed_x, double observed_y, string cam_name):observed_x(observed_x), observed_y(observed_y), cam_name(cam_name) {}

    template <typename T>
    //bool operator()(const T* const camera, const T* const point, T* residuals) const {
    bool operator()(const T* const point, const T* const poseCam1ToCam_i, T* residuals) const {

        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        //T p_tmp[3];
        //ceres::AngleAxisRotatePoint(cam01_pose, point, p_tmp);

        // camera[3,4,5] are the translation.
        //p_tmp[0] += cam01_pose[3];
        //p_tmp[1] += cam01_pose[4];
        //p_tmp[2] += cam01_pose[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        ceres::AngleAxisRotatePoint(poseCam1ToCam_i, point, p);
        p[0] += poseCam1ToCam_i[3];
        p[1] += poseCam1ToCam_i[4];
        p[2] += poseCam1ToCam_i[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Apply second and fourth order radial distortion.
        // Compute final projected point position.
        T predicted_x = K[cam_name][0]*xp + K[cam_name][2]; //focal * distortion * xp;    ##bug!!!
        T predicted_y = K[cam_name][1]*yp + K[cam_name][3]; //focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x, const double observed_y, const string camera_name) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Cam_else, 2, 3, 6>( //2,3,6,6 最后一个数字，6表示优化poseCam1ToCam_i的R+T，3表示只优化R
                new SnavelyReprojectionError_Cam_else(observed_x, observed_y, camera_name)));
    }

    double observed_x;
    double observed_y;
    string cam_name;
};

int main(int argc, char** argv) {
    /*
    cout<<"debug"<<endl;
    video_num = argv[1];
    string path = "/data/disk_a/xujl/Project/FSDE_new/tools/LoFTR/tmp/@/points.txt";
    path = path.replace(path.find("@"),1,video_num);
    cout<<path<<endl;
    ifstream myfile_debug(path);
    double a;
    for(int i=0;i<10;i++){
            myfile_debug>>a;
            cout<<a<<" ";
        }
    myfile_debug.close();
    */

    video_num = argv[1];

    readFileJson();
    google::InitGoogleLogging(argv[0]);
    BALProblem bal_problem;
    bal_problem.LoadFile();

    const double* observations = bal_problem.observations();

    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
        //int camera_index_i = cameras_index[i];
        int point_index_i = (i-i%2)/2;
        string cam_name = camera_index2[i].cam;//相同是i是因为 obsevation 和 camera_index 的顺序相同
        string timestamp = camera_index2[i].timestamp;
        if(cam_name=="L"){
            //continue;
            ceres::CostFunction* cost_function =
                SnavelyReprojectionError_Cam_01::Create(observations[2 * i + 0],
                                             observations[2 * i + 1],
                                             cam_name
                                             );

            problem.AddResidualBlock(cost_function,
                                 NULL, // ##squared loss#
                                 bal_problem.mutable_point_for_observation(point_index_i)
                                 //bal_problem.mutable_cam01_pose(timestamp)
                                 );
            }
        else{
            //continue;
            ceres::CostFunction* cost_function =
                SnavelyReprojectionError_Cam_else::Create(observations[2 * i + 0],
                                                observations[2 * i + 1],
                                                cam_name
                                                );
            problem.AddResidualBlock(cost_function,
                             NULL, // ##squared loss##
                             bal_problem.mutable_point_for_observation(point_index_i),
                             //bal_problem.mutable_cam01_pose(timestamp),
                             bal_problem.mutable_Pose_Cam01ToCam_i(cam_name)
                             );
        }

    }
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.

    if (true){
        ceres::Solver::Options options;
        options.max_num_iterations=5000;
        //options.use_explicit_schur_complement = true;
        options.num_threads=60;
        //options.linear_solver_type = ceres::DENSE_SCHUR;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
    }

    ofstream outf;
    path = root_path + video_num;
    outf.open(path + "/parameters.txt");
    //cout<<"num="<<Num_relative_cam<<" " <<Num_timestamp<<" "<<Num_points<<endl;
    for(int i=0;i<(Num_relative_cam+Num_timestamp)*6+Num_points*3;i++){
        outf<<i<<" = "<<*(bal_problem.mutable_cameras()+i)<<endl;
    }
    outf.close();
    cout<<"write done"<<endl;
    return 0;
}
