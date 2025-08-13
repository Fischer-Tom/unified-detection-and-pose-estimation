import numpy as np
from skimage.measure import ransac
from scipy.optimize import least_squares
from einops import einsum
import cv2
from scipy.spatial import ConvexHull

from src.uni_dp.tools import convert_points_py3d_to_cv2

class PoseRefiner:
    def __init__(self, downsample_rate=4):
        self.sizes = None
        self.Rs = None
        self.Ts = None
        self.pred_labels = None
        self.downsample_rate = downsample_rate



    def get_params(self):
        return [(label, R, T, size) for label, R, T, size in
                zip(self.pred_labels, self.Rs, self.Ts, self.sizes)]



    def run(self, repr_err_thr: float=4.0):
        sizes = []
        Rs = []
        Ts = []
        indices_to_delete = []
        for i in range(self.n_objects):

            # Run RANSAC for each object
            if len(self.args[i][0]) < 6:
                print(f"Not enough points for object {i}, skipping RANSAC.")
                indices_to_delete.append(i)
                continue
            # Use RANSAC to estimate the size of the object
            best_model, inliers = ransac(self.args[i], DeformationModel, min_samples=6,
                                   residual_threshold=repr_err_thr, max_trials=100, stop_probability=0.999)
            if best_model is None or (sum(inliers) / len(self.args[i][0])) < 0.1:
                print(f"RANSAC failed for object {i}, using default size.")
                indices_to_delete.append(i)
                continue
            else:
                size = best_model.params
            # Run PnP for each object
            success, r, T, inliers = cv2.solvePnPRansac(
                (self.args[i][0] * size).astype(np.float64),
                self.args[i][1].astype(np.float64),
                self.args[i][4][0].astype(np.float64),
                rvec=self.args[i][2][0].astype(np.float64),
                tvec=self.args[i][3][0].astype(np.float64),
                useExtrinsicGuess=True,
                distCoeffs=None,
                reprojectionError=repr_err_thr,
                flags=cv2.SOLVEPNP_ITERATIVE,
                iterationsCount=200,
            )
            R, _ = cv2.Rodrigues(r)

            sizes.append(size)
            Rs.append(R)
            Ts.append(T)
        if len(indices_to_delete) > 0:
            for index in sorted(indices_to_delete, reverse=True):
                del self.pred_labels[index]
        self.sizes = sizes
        self.Rs = Rs
        self.Ts = Ts


    def setup(self, corr_dict, m_poses, K, im_shape):
        self.args = self._process_data(corr_dict, m_poses, K, im_shape)

    def _process_data(self, corr_dict, m_poses, K, im_shape):
        K = K.squeeze().cpu().numpy()
        K[:2] /= self.downsample_rate
        points3d, points2d, pred_rs, pred_ts, cls_pred  = [], [], [], [], []
        for pose in m_poses:
            label, R, T, corr2d = pose
            mask = np.zeros((im_shape[0] // self.downsample_rate, im_shape[1] // self.downsample_rate), dtype=np.bool)
            mask[corr2d[:,0], corr2d[:,1]] = True
            obj_mask = mask.reshape(-1) * corr_dict["mask"].cpu().numpy()
            o_p = corr_dict["corr3d"].reshape(-1, 3).cpu().numpy()[obj_mask]
            i_p = corr_dict["corr2d"].reshape(-1, 2).cpu().numpy()[obj_mask]
            points3d.append(o_p)
            points2d.append(i_p)
            pred_rs.append(cv2.Rodrigues(R)[0])
            pred_ts.append(T)
            cls_pred.append(label)


        args = [(*convert_points_py3d_to_cv2(op, ip),
                 R[None,...].repeat(max(op.shape[0],1), 0),
                 T[None,...].repeat(max(op.shape[0],1), 0),
                 K[None,...].repeat(max(op.shape[0],1), 0))
                for op, ip, R, T in zip(points3d, points2d, pred_rs, pred_ts)]


        self.K = K
        self.Rs = pred_rs
        self.Ts = pred_ts
        self.pred_labels = cls_pred
        self.n_objects = len(args)
        self.im_size = (im_shape[0] // self.downsample_rate, im_shape[1] // self.downsample_rate)
        return args


class DeformationModel:
    def __init__(self):
        self.params = np.ones(3, dtype=np.float32)

    def residuals(self, op, ip, R, T, K):
        return self.residual_fn(self.params, op, ip, R, T, K)


    def estimate(self, op, ip, R, T, K):
        if len(op) < 6:
            return False
        res = least_squares(fun=self.residual_fn, method="lm",
                            x0=self.params,
                            args=(op, ip, R, T, K),
                            verbose=0,
                            xtol=1e-6)
        self.params = res.x
        return True


    def residual_fn(self, x, objectPoints, imagePoints, R, T, K):
        R = cv2.Rodrigues(R[0])[0]
        scaled_points = x[None,:] * objectPoints

        transformed_points = einsum(
            K[0],
            einsum(R, scaled_points, "i j, v j -> v i ") + T,
            "i j, v j -> v i",
        )

        transformed_points = (
                transformed_points[..., :2] / transformed_points[:, -1, None]
        )

        loss = np.linalg.norm(transformed_points - imagePoints, axis=-1)
        return loss.flatten()