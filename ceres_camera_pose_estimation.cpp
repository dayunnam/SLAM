
/*
https://github.com/izhengfan/ba_demo_ceres
*/

#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdlib>


#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>


using namespace Eigen;
using namespace std;

const double SMALL_EPS = 1e-10;
typedef Eigen::Matrix<double, 6, 1, Eigen::ColMajor> Vector6d;
typedef Eigen::Matrix<double, 7, 1, Eigen::ColMajor> Vector7d;
constexpr int USE_POSE_SIZE = 6;


static double uniform_rand(double lowerBndr, double upperBndr)
{
    return lowerBndr + ((double)std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}


static double gauss_rand(double mean, double sigma)
{
    double x, y, r2;
    do
    {
        x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0.0);
    return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}



inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    m.fill(0.);
    m(0, 1) = -v(2);
    m(0, 2) = v(1);
    m(1, 2) = -v(0);
    m(1, 0) = v(2);
    m(2, 0) = -v(1);
    m(2, 1) = v(0);
    return m;
}

inline Eigen::Vector3d deltaR(const Eigen::Matrix3d& R)
{
    Eigen::Vector3d v;
    v(0) = R(2, 1) - R(1, 2);
    v(1) = R(0, 2) - R(2, 0);
    v(2) = R(1, 0) - R(0, 1);
    return v;
}


inline Eigen::Vector3d toAngleAxis(const Eigen::Quaterniond& quaterd, double* angle = NULL)
{
    Eigen::Quaterniond unit_quaternion = quaterd.normalized();
    double n = unit_quaternion.vec().norm();
    double w = unit_quaternion.w();
    double squared_w = w * w;

    double two_atan_nbyw_by_n;
    // Atan-based log thanks to
    //
    // C. Hertzberg et al.:
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011

    if (n < SMALL_EPS)
    {
        // If quaternion is normalized and n=1, then w should be 1;
        // w=0 should never happen here!
        assert(fabs(w) > SMALL_EPS);

        two_atan_nbyw_by_n = 2. / w - 2. * (n * n) / (w * squared_w);
    }
    else
    {
        if (fabs(w) < SMALL_EPS)
        {
            if (w > 0)
            {
                two_atan_nbyw_by_n = M_PI / n;
            }
            else
            {
                two_atan_nbyw_by_n = -M_PI / n;
            }
        }
        two_atan_nbyw_by_n = 2 * atan(n / w) / n;
    }
    if (angle != NULL) *angle = two_atan_nbyw_by_n * n;
    return two_atan_nbyw_by_n * unit_quaternion.vec();
}

inline Eigen::Quaterniond toQuaterniond(const Eigen::Vector3d& v3d, double* angle = NULL)
{
    double theta = v3d.norm();
    if (angle != NULL)
        *angle = theta;
    double half_theta = 0.5 * theta;

    double imag_factor;
    double real_factor = cos(half_theta);
    if (theta < SMALL_EPS)
    {
        double theta_sq = theta * theta;
        double theta_po4 = theta_sq * theta_sq;
        imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
    }
    else
    {
        double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta / theta;
    }

    return Eigen::Quaterniond(real_factor,
        imag_factor * v3d.x(),
        imag_factor * v3d.y(),
        imag_factor * v3d.z());
}


class SE3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

protected:

    Eigen::Quaterniond _r;
    Eigen::Vector3d _t;

public:
    SE3() {
        _r.setIdentity();
        _t.setZero();
    }

    SE3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) :_r(Eigen::Quaterniond(R)), _t(t) {
        normalizeRotation();
    }

    SE3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) :_r(q), _t(t) {
        normalizeRotation();
    }

    inline const Eigen::Vector3d& translation() const { return _t; }

    inline Eigen::Vector3d& translation() { return _t; }

    inline void setTranslation(const Eigen::Vector3d& t_) { _t = t_; }

    inline const Eigen::Quaterniond& rotation() const { return _r; }

    inline Eigen::Quaterniond& rotation() { return _r; }

    void setRotation(const Eigen::Quaterniond& r_) { _r = r_; }

    inline SE3 operator* (const SE3& tr2) const {
        SE3 result(*this);
        result._t += _r * tr2._t;
        result._r *= tr2._r;
        result.normalizeRotation();
        return result;
    }

    inline SE3& operator*= (const SE3& tr2) {
        _t += _r * tr2._t;
        _r *= tr2._r;
        normalizeRotation();
        return *this;
    }

    inline Eigen::Vector3d operator* (const Eigen::Vector3d& v) const {
        return _t + _r * v;
    }

    inline SE3 inverse() const {
        SE3 ret;
        ret._r = _r.conjugate();
        ret._t = ret._r * (_t * -1.);
        return ret;
    }

    inline double operator [](int i) const {
        assert(i < 7);
        if (i < 4)
            return _r.coeffs()[i];
        return _t[i - 4];
    }


    inline Vector7d toVector() const {
        Vector7d v;
        v.head<4>() = Eigen::Vector4d(_r.coeffs());
        v.tail<3>() = _t;
        return v;
    }

    inline void fromVector(const Vector7d& v) {
        _r = Eigen::Quaterniond(v[3], v[0], v[1], v[2]);
        _t = Eigen::Vector3d(v[4], v[5], v[6]);
    }


    Vector6d log() const {
        Vector6d res;

        double theta;
        res.head<3>() = toAngleAxis(_r, &theta);

        Eigen::Matrix3d Omega = skew(res.head<3>());
        Eigen::Matrix3d V_inv;
        if (theta < SMALL_EPS)
        {
            V_inv = Eigen::Matrix3d::Identity() - 0.5 * Omega + (1. / 12.) * (Omega * Omega);
        }
        else
        {
            V_inv = (Eigen::Matrix3d::Identity() - 0.5 * Omega
                + (1 - theta / (2 * tan(theta / 2))) / (theta * theta) * (Omega * Omega));
        }

        res.tail<3>() = V_inv * _t;

        return res;
    }

    Eigen::Vector3d map(const Eigen::Vector3d& xyz) const
    {
        return _r * xyz + _t;
    }


    static SE3 exp(const Vector6d& update)
    {
        Eigen::Vector3d omega(update.data());
        Eigen::Vector3d upsilon(update.data() + 3);

        double theta;
        Eigen::Matrix3d Omega = skew(omega);

        Eigen::Quaterniond R = toQuaterniond(omega, &theta);
        Eigen::Matrix3d V;
        if (theta < SMALL_EPS)
        {
            V = R.matrix();
        }
        else
        {
            Eigen::Matrix3d Omega2 = Omega * Omega;

            V = (Eigen::Matrix3d::Identity()
                + (1 - cos(theta)) / (theta * theta) * Omega
                + (theta - sin(theta)) / (pow(theta, 3)) * Omega2);
        }
        return SE3(R, V * upsilon);
    }

    Eigen::Matrix<double, 6, 6, Eigen::ColMajor> adj() const
    {
        Eigen::Matrix3d R = _r.toRotationMatrix();
        Eigen::Matrix<double, 6, 6, Eigen::ColMajor> res;
        res.block(0, 0, 3, 3) = R;
        res.block(3, 3, 3, 3) = R;
        res.block(3, 0, 3, 3) = skew(_t) * R;
        res.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero(3, 3);
        return res;
    }

    Eigen::Matrix<double, 4, 4, Eigen::ColMajor> to_homogeneous_matrix() const
    {
        Eigen::Matrix<double, 4, 4, Eigen::ColMajor> homogeneous_matrix;
        homogeneous_matrix.setIdentity();
        homogeneous_matrix.block(0, 0, 3, 3) = _r.toRotationMatrix();
        homogeneous_matrix.col(3).head(3) = translation();

        return homogeneous_matrix;
    }

    void normalizeRotation() {
        if (_r.w() < 0) {
            _r.coeffs() *= -1;
        }
        _r.normalize();
    }
};



class CameraParameters
{
protected:
    double f;
    double cx;
    double cy;
public:
    CameraParameters(double f_, double cx_, double cy_)
        : f(f_), cx(cx_), cy(cy_) {}

    Vector2d cam_map(const Vector3d& p)
    {
        Vector2d z;
        z[0] = f * p[0] / p[2] + cx;
        z[1] = f * p[1] / p[2] + cy;
        return z;
    }
};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class ReprojectionErrorSE3XYZ : public ceres::SizedCostFunction<2, PoseBlockSize, 3>
{
public:
    ReprojectionErrorSE3XYZ(double f_,
        double cx_,
        double cy_,
        double observation_x,
        double observation_y)
        : f(f_), cx(cx_), cy(cy_),
        _observation_x(observation_x),
        _observation_y(observation_y) {}

    virtual bool Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const;

    double f;
    double cx;
    double cy;

private:
    double _observation_x;
    double _observation_y;
};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class PoseSE3Parameterization : public ceres::LocalParameterization {
public:
    PoseSE3Parameterization() {}
    virtual ~PoseSE3Parameterization() {}
    virtual bool Plus(const double* x,
        const double* delta,
        double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
        double* jacobian) const;
    virtual int GlobalSize() const { return PoseBlockSize; }
    virtual int LocalSize() const { return 6; }
};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class PosePointParametersBlock
{
public:
    PosePointParametersBlock() {}
    void create(int pose_num, int point_num)
    {
        poseNum = pose_num;
        pointNum = point_num;
        values = new double[pose_num * PoseBlockSize + point_num * 3];
    }
    PosePointParametersBlock(int pose_num, int point_num) : poseNum(pose_num), pointNum(point_num)
    {
        values = new double[pose_num * PoseBlockSize + point_num * 3];
    }
    ~PosePointParametersBlock() { delete[] values; }

    void setPose(int idx, const Quaterniond& q, const Vector3d& trans);

    void getPose(int idx, Quaterniond& q, Vector3d& trans);

    double* pose(int idx) { return values + idx * PoseBlockSize; }

    double* point(int idx) { return values + poseNum * PoseBlockSize + idx * 3; }

    int poseNum;
    int pointNum;
    double* values;

};

template<>
bool ReprojectionErrorSE3XYZ<7>::Evaluate(const double* const* parameters, double* residuals, double** jacobians) const
{
    Eigen::Map<const Quaterniond> quaterd(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    Eigen::Vector3d p = quaterd * point + trans;

    double f_by_z = f / p[2];
    residuals[0] = f_by_z * p[0] + cx - _observation_x;
    residuals[1] = f_by_z * p[1] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_zz = f_by_z / p[2];
    J_cam << f_by_z, 0, -f_by_zz * p[0],
        0, f_by_z, -f_by_zz * p[1];


    if (jacobians != NULL)
    {
        if (jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();
            J_se3.block<2, 3>(0, 0) = -J_cam * skew(p);
            J_se3.block<2, 3>(0, 3) = J_cam;
        }
        if (jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[1]);
            J_point = J_cam * quaterd.toRotationMatrix();
        }
    }

    return true;
}

template<>
bool ReprojectionErrorSE3XYZ<6>::Evaluate(const double* const* parameters, double* residuals, double** jacobians) const
{
    Eigen::Quaterniond quaterd = toQuaterniond(Eigen::Map<const Vector3d>(parameters[0]));
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 3);
    Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    Eigen::Vector3d p = quaterd * point + trans;

    double f_by_z = f / p[2];
    residuals[0] = f_by_z * p[0] + cx - _observation_x;
    residuals[1] = f_by_z * p[1] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_zz = f_by_z / p[2];
    J_cam << f_by_z, 0, -f_by_zz * p[0],
        0, f_by_z, -f_by_zz * p[1];

    if (jacobians != NULL)
    {
        if (jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.block<2, 3>(0, 0) = -J_cam * skew(p);
            J_se3.block<2, 3>(0, 3) = J_cam;
        }
        if (jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[1]);
            J_point = J_cam * quaterd.toRotationMatrix();
        }
    }

    return true;
}


template<>
bool PoseSE3Parameterization<7>::Plus(const double* x, const double* delta, double* x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> trans(x + 4);
    SE3 se3_delta = SE3::exp(Eigen::Map<const Vector6d>(delta));

    Eigen::Map<const Eigen::Quaterniond> quaterd(x);
    Eigen::Map<Eigen::Quaterniond> quaterd_plus(x_plus_delta);
    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);

    quaterd_plus = se3_delta.rotation() * quaterd;
    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();

    return true;
}

template<>
bool PoseSE3Parameterization<7>::ComputeJacobian(const double* x, double* jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > J(jacobian);
    J.setZero();
    J.block<6, 6>(0, 0).setIdentity();
    return true;
}



template<>
bool PoseSE3Parameterization<6>::Plus(const double* x, const double* delta, double* x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> trans(x + 3);
    SE3 se3_delta = SE3::exp(Eigen::Map<const Vector6d>(delta));

    Quaterniond quaterd_plus = se3_delta.rotation() * toQuaterniond(Eigen::Map<const Vector3d>(x));
    Eigen::Map<Vector3d> angles_plus(x_plus_delta);
    angles_plus = toAngleAxis(quaterd_plus);

    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 3);
    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();
    return true;
}

template<>
bool PoseSE3Parameterization<6>::ComputeJacobian(const double* x, double* jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J(jacobian);
    J.setIdentity();
    return true;
}



template<>
void PosePointParametersBlock<7>::getPose(int idx, Quaterniond& q, Vector3d& trans)
{
    double* pose_ptr = values + idx * 7;
    q = Map<const Quaterniond>(pose_ptr);
    trans = Map<const Vector3d>(pose_ptr + 4);
}

template<>
void PosePointParametersBlock<7>::setPose(int idx, const Quaterniond& q, const Vector3d& trans)
{
    double* pose_ptr = values + idx * 7;
    Eigen::Map<Vector7d> pose(pose_ptr);
    pose.head<4>() = Eigen::Vector4d(q.coeffs());
    pose.tail<3>() = trans;
}

template<>
void PosePointParametersBlock<6>::getPose(int idx, Quaterniond& q, Vector3d& trans)
{
    double* pose_ptr = values + idx * 6;
    q = toQuaterniond(Vector3d(pose_ptr));
    trans = Map<const Vector3d>(pose_ptr + 3);
}

template<>
void PosePointParametersBlock<6>::setPose(int idx, const Quaterniond& q, const Vector3d& trans)
{
    double* pose_ptr = values + idx * 6;
    Eigen::Map<Vector6d> pose(pose_ptr);
    pose.head<3>() = toAngleAxis(q);
    pose.tail<3>() = trans;
}
class Sample
{
public:
    static int uniform(int from, int to)
    {
        return static_cast<int>(uniform_rand(from, to));
    }

    static double uniform()
    {
        return uniform_rand(0., 1.);
    }
    static double gaussian(double sigma)
    {
        return gauss_rand(0., sigma);
    }

};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template <int PoseBlockSize>
class BAProblem
{
public:
    BAProblem(int pose_num_, int point_num_, double pix_noise_, bool useOrdering = false);

    void solve(ceres::Solver::Options& opt, ceres::Solver::Summary* sum);

    ceres::Problem problem;
    ceres::ParameterBlockOrdering* ordering = NULL;

protected:
    PosePointParametersBlock<PoseBlockSize> states;
    PosePointParametersBlock<PoseBlockSize> true_states;

};

template<int PoseBlockSize>
BAProblem<PoseBlockSize>::BAProblem(int pose_num_, int point_num_, double pix_noise_, bool useOrdering)
{
    if (useOrdering)
        ordering = new ceres::ParameterBlockOrdering;

    int pose_num = pose_num_;
    int point_num = point_num_;
    double PIXEL_NOISE = pix_noise_;

    states.create(pose_num, point_num);
    true_states.create(pose_num, point_num);

    for (int i = 0; i < point_num; ++i)
    {
        Eigen::Map<Vector3d> true_pt(true_states.point(i));
        true_pt = Vector3d((Sample::uniform() - 0.5) * 3,
            Sample::uniform() - 0.5,
            Sample::uniform() + 3);
    }

    double focal_length = 1000.;
    double cx = 320.;
    double cy = 240.;
    CameraParameters cam(focal_length, cx, cy);

    for (int i = 0; i < pose_num; ++i)
    {
        Vector3d trans(i * 0.04 - 1., 0, 0);

        Eigen::Quaterniond q;
        q.setIdentity();
        true_states.setPose(i, q, trans);
        states.setPose(i, q, trans);

        problem.AddParameterBlock(states.pose(i), PoseBlockSize, new PoseSE3Parameterization<PoseBlockSize>());

        if (i < 2)
        {
            problem.SetParameterBlockConstant(states.pose(i));
        }
    }

    for (int i = 0; i < point_num; ++i)
    {
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        Eigen::Map<Vector3d> noise_point_i(states.point(i));
        noise_point_i = true_point_i + Vector3d(Sample::gaussian(1),
            Sample::gaussian(1),
            Sample::gaussian(1));

        Vector2d z;
        SE3 true_pose_se3;

        int num_obs = 0;
        for (int j = 0; j < pose_num; ++j)
        {
            true_states.getPose(j, true_pose_se3.rotation(), true_pose_se3.translation());
            Vector3d point_cam = true_pose_se3.map(true_point_i);
            z = cam.cam_map(point_cam);
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
            {
                ++num_obs;
            }
        }
        if (num_obs >= 2)
        {
            problem.AddParameterBlock(states.point(i), 3);
            if (useOrdering)
                ordering->AddElementToGroup(states.point(i), 0);

            for (int j = 0; j < pose_num; ++j)
            {
                true_states.getPose(j, true_pose_se3.rotation(), true_pose_se3.translation());
                Vector3d point_cam = true_pose_se3.map(true_point_i);
                z = cam.cam_map(point_cam);

                if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
                {
                    z += Vector2d(Sample::gaussian(PIXEL_NOISE),
                        Sample::gaussian(PIXEL_NOISE));

                    ceres::CostFunction* costFunc = new ReprojectionErrorSE3XYZ<PoseBlockSize>(focal_length, cx, cy, z[0], z[1]);
                    problem.AddResidualBlock(costFunc, NULL, states.pose(j), states.point(i));
                }
            }

        }
    }

    if (useOrdering)
        for (int i = 0; i < pose_num; ++i)
        {
            ordering->AddElementToGroup(states.pose(i), 1);
        }

}


template<int PoseBlockSize>
void BAProblem<PoseBlockSize>::solve(ceres::Solver::Options& opt, ceres::Solver::Summary* sum)
{
    if (ordering != NULL)
        opt.linear_solver_ordering.reset(ordering);
    ceres::Solve(opt, &problem, sum);
}

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        cout << endl;
        cout << "Please type: " << endl;
        cout << "ba_demo [PIXEL_NOISE] " << endl;
        cout << endl;
        cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
        cout << endl;
        exit(0);
    }
    
    google::InitGoogleLogging(argv[0]);

    double PIXEL_NOISE = atof(argv[1]);

    cout << "PIXEL_NOISE: " << PIXEL_NOISE << endl;

    BAProblem<USE_POSE_SIZE> baProblem(15, 300, PIXEL_NOISE, true);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    baProblem.solve(options, &summary);
    std::cout << summary.BriefReport() << "\n";
}
