import numpy as np

'Discrete Time'
Ts = 0.02


class TinyCar_model:

    def __init__(self, sigma):
        self.Cm1 = 0.287
        self.Cm2 = 0.0545
        self.Cr0 = 0.0518
        self.Cr2 = 0.00035

        self.Df = 0.192
        self.Cf = 1.2
        self.Bf = 2.579

        self.Dr = 0.1737
        self.Cr = 1.2691
        self.Br = 3.3852

        self.lf = 0.029
        self.lr = 0.033

        self.m = 0.041
        self.I = 27.8E-6

        self.Sv = 0.01

        self.max_steering = 0.35
        self.max_throttle = 1

        self.max_speed = 3.5

        self.min_vx = 0.01
        self.max_vx = 3.8
        self.min_vy = -3.2
        self.max_vy = 3.2
        self.min_r = -9
        self.max_r = 9

        self.sigma = sigma

    def update(self, vx, vy, r, throttle, steering):

        # add noise
        vx = vx * (1 + np.random.normal() * self.sigma)
        vy = vy * (1 + np.random.normal() * self.sigma)
        r = r * (1 + np.random.normal() * self.sigma)

        delta = steering * self.max_steering
        d = throttle * self.max_throttle

        alpha_f = - np.arctan2(vy + r * self.lf, vx) + delta
        fy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f)) + self.Sv
        alpha_r = - np.arctan2(vy - r * self.lr, vx)
        ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r)) + self.Sv
        rx = self.Cm1 * d - self.Cm2 * d * vx

        # f = - self.Cr2 * vx * vx
        f = -self.Cr0 - self.Cr2 * vx * vx

        dvx = 1./self.m * (rx + f - fy * np.sin(delta) + self.m * vy * r)
        dvy = 1./self.m * (ry + fy * np.cos(delta) - self.m * vx * r)
        dr = 1./self.I * (self.lf * fy * np.cos(delta) - self.lr * ry)

        vx = min(max(vx + dvx * Ts, self.min_vx), self.max_vx)
        vy = min(max(vy + dvy * Ts, self.min_vy), self.max_vy)
        r = min(max(r + dr * Ts, self.min_r), self.max_r)

        return vx, vy, r, dvx, dvy, dr


class TinyCar_model_error:

    def __init__(self):
        self.Cm1 = 0.301
        self.Cm2 = 0.0572
        self.Cr0 = 0.05439
        self.Cr2 = 0.0005

        self.Df = 0.22
        self.Cf = 1.3
        self.Bf = 3

        self.Dr = 0.2
        self.Cr = 1.35
        self.Br = 3.8

        self.lf = 0.029
        self.lr = 0.033

        self.m = 0.041
        self.I = 33.58E-6

        self.max_steering = 0.35
        self.max_throttle = 1

        self.max_speed = 3.5

        self.min_vx = 0.01
        self.max_vx = 3.8
        self.min_vy = -3.2
        self.max_vy = 3.2
        self.min_r = -9
        self.max_r = 9

    def update(self, vx, vy, r, throttle, steering):

        delta = steering * self.max_steering
        d = throttle * self.max_throttle

        alpha_f = - np.arctan2(vy + r * self.lf, vx) + delta
        fy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f))
        alpha_r = - np.arctan2(vy - r * self.lr, vx)
        ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r))
        rx = self.Cm1 * d - self.Cm2 * d * vx

        # f = - self.Cr2 * vx * vx
        f = -self.Cr0 - self.Cr2 * vx * vx

        dvx = 1./self.m * (rx + f - fy * np.sin(delta) + self.m * vy * r)
        dvy = 1./self.m * (ry + fy * np.cos(delta) - self.m * vx * r)
        dr = 1./self.I * (self.lf * fy * np.cos(delta) - self.lr * ry)

        vx = min(max(vx + dvx * Ts, self.min_vx), self.max_vx)
        vy = min(max(vy + dvy * Ts, self.min_vy), self.max_vy)
        r = min(max(r + dr * Ts, self.min_r), self.max_r)

        return vx, vy, r, dvx, dvy, dr


class FormulaCar_model:

    def __init__(self):
        self.Cm1 = 8000
        self.Cm2 = 172

        self.Cr0 = 180
        self.Cr2 = 0.7

        self.Br = 10
        self.Cr = 1.38
        self.Dr = 1500

        self.Bf = 10
        self.Cf = 1.38
        self.Df = 1500

        self.lf = 1.22
        self.lr = 1.22

        self.m = 190
        self.I = 110

        self.car_l = 2.0
        self.car_w = 1.0

        self.max_steering = 0.5
        self.max_throttle = 1

        self.max_speed = 3.5

        self.min_vx = 0.02
        self.max_vx = 33
        self.min_vy = -10.5
        self.max_vy = 10.5
        self.min_r = -2.2
        self.max_r = 2.2

    def update(self, vx, vy, r, throttle, steering):

        delta = steering * self.max_steering
        d = throttle * self.max_throttle

        alpha_f = - np.arctan2(vy + r * self.lf, vx) + delta
        fy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f))
        alpha_r = - np.arctan2(vy - r * self.lr, vx)
        ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r))
        rx = self.Cm1 * d - self.Cm2 * d * vx

        # f = - self.Cr2 * vx * vx
        f = -self.Cr0 - self.Cr2 * vx * vx

        dvx = 1./self.m * (rx + f - fy * np.sin(delta) + self.m * vy * r)
        dvy = 1./self.m * (ry + fy * np.cos(delta) - self.m * vx * r)
        dr = 1./self.I * (self.lf * fy * np.cos(delta) - self.lr * ry)

        vx = min(max(vx + dvx * Ts, self.min_vx), self.max_vx)
        vy = min(max(vy + dvy * Ts, self.min_vy), self.max_vy)
        r = min(max(r + dr * Ts, self.min_r), self.max_r)

        return vx, vy, r, dvx, dvy, dr


class FormulaCar_model_error5:

    def __init__(self):
        self.Cm1 = 8400
        self.Cm2 = 181

        self.Cr0 = 189
        self.Cr2 = 0.74

        self.Br = 10.5
        self.Cr = 1.45
        self.Dr = 1575

        self.Bf = 10.5
        self.Cf = 1.45
        self.Df = 1575

        self.lf = 1.22
        self.lr = 1.22

        self.m = 190
        self.I = 110

        self.car_l = 2.0
        self.car_w = 1.0

        self.max_steering = 0.5
        self.max_throttle = 1

        self.max_speed = 3.5

        self.min_vx = 0.02
        self.max_vx = 33
        self.min_vy = -10.5
        self.max_vy = 10.5
        self.min_r = -2.2
        self.max_r = 2.2

    def update(self, vx, vy, r, throttle, steering):

        delta = steering * self.max_steering
        d = throttle * self.max_throttle

        alpha_f = - np.arctan2(vy + r * self.lf, vx) + delta
        fy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f))
        alpha_r = - np.arctan2(vy - r * self.lr, vx)
        ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r))
        rx = self.Cm1 * d - self.Cm2 * d * vx

        # f = - self.Cr2 * vx * vx
        f = -self.Cr0 - self.Cr2 * vx * vx

        dvx = 1./self.m * (rx + f - fy * np.sin(delta) + self.m * vy * r)
        dvy = 1./self.m * (ry + fy * np.cos(delta) - self.m * vx * r)
        dr = 1./self.I * (self.lf * fy * np.cos(delta) - self.lr * ry)

        vx = min(max(vx + dvx * Ts, self.min_vx), self.max_vx)
        vy = min(max(vy + dvy * Ts, self.min_vy), self.max_vy)
        r = min(max(r + dr * Ts, self.min_r), self.max_r)

        return vx, vy, r, dvx, dvy, dr


class FormulaCar_model_error10:

    def __init__(self):
        self.Cm1 = 8800
        self.Cm2 = 190

        self.Cr0 = 198
        self.Cr2 = 0.77

        self.Br = 11
        self.Cr = 1.53
        self.Dr = 1650

        self.Bf = 11
        self.Cf = 1.53
        self.Df = 1650

        self.lf = 1.22
        self.lr = 1.22

        self.m = 190
        self.I = 110

        self.car_l = 2.0
        self.car_w = 1.0

        self.max_steering = 0.5
        self.max_throttle = 1

        self.max_speed = 3.5

        self.min_vx = 0.02
        self.max_vx = 33
        self.min_vy = -10.5
        self.max_vy = 10.5
        self.min_r = -2.2
        self.max_r = 2.2

    def update(self, vx, vy, r, throttle, steering):

        delta = steering * self.max_steering
        d = throttle * self.max_throttle

        alpha_f = - np.arctan2(vy + r * self.lf, vx) + delta
        fy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f))
        alpha_r = - np.arctan2(vy - r * self.lr, vx)
        ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r))
        rx = self.Cm1 * d - self.Cm2 * d * vx

        # f = - self.Cr2 * vx * vx
        f = -self.Cr0 - self.Cr2 * vx * vx

        dvx = 1./self.m * (rx + f - fy * np.sin(delta) + self.m * vy * r)
        dvy = 1./self.m * (ry + fy * np.cos(delta) - self.m * vx * r)
        dr = 1./self.I * (self.lf * fy * np.cos(delta) - self.lr * ry)

        vx = min(max(vx + dvx * Ts, self.min_vx), self.max_vx)
        vy = min(max(vy + dvy * Ts, self.min_vy), self.max_vy)
        r = min(max(r + dr * Ts, self.min_r), self.max_r)

        return vx, vy, r, dvx, dvy, dr


class zonda_model:

    def __init__(self):
        self.Cm1 = 0.287
        self.Cm2 = 0.0545
        self.Cr0 = 0.0518
        self.Cr2 = 0.00035

        self.Df = 0.192
        self.Cf = 1.2
        self.Bf = 2.579

        self.Dr = 0.1737
        self.Cr = 1.2691
        self.Br = 3.3852

        self.lf = 0.13
        self.lr = 0.12

        self.m = 2.17
        self.Iz = 9.837e-3

        self.max_steering = 0.35
        self.max_throttle = 1

        self.max_speed = 3.5

        self.min_vx = 0.01
        self.max_vx = 3.8
        self.min_vy = -3.2
        self.max_vy = 3.2
        self.min_r = -9
        self.max_r = 9

    def update(self, vx, vy, r, throttle, steering):

        delta = steering * self.max_steering
        d = throttle * self.max_throttle

        alpha_f = - np.arctan2(vy + r * self.lf, vx) + delta
        fy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f))
        alpha_r = - np.arctan2(vy - r * self.lr, vx)
        ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r))
        rx = self.Cm1 * d - self.Cm2 * d * vx

        # f = - self.Cr2 * vx * vx
        f = -self.Cr0 - self.Cr2 * vx * vx

        dvx = 1./self.m * (rx + f - fy * np.sin(delta) + self.m * vy * r)
        dvy = 1./self.m * (ry + fy * np.cos(delta) - self.m * vx * r)
        dr = 1./self.I * (self.lf * fy * np.cos(delta) - self.lr * ry)

        vx = min(max(vx + dvx * Ts, self.min_vx), self.max_vx)
        vy = min(max(vy + dvy * Ts, self.min_vy), self.max_vy)
        r = min(max(r + dr * Ts, self.min_r), self.max_r)

        return vx, vy, r, dvx, dvy, dr

