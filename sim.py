import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

g = 9.80665
# maxmimum thrust of each rotor
thrustMax = 7


# keep thrust in reasonable range
def _thresholdControl(thrust):

    return max(0, min(thrust, thrustMax))


# gravitational acceleration in ground frame
gravity = [0, 0, - g]
# mass of the drone
mass = 1.5
# distance between the center of the drone and each rotor
d = 0.5
# constant ratio in relation between angular speed of rotor
# and the thrust it produces
# f = k * omega**2
rotorAngularSpeedConstant = 0.01


class Velocities:
    def __init__(self, vx, vy, vz, vpsi, vtheta, vphi):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.vpsi = vpsi
        self.vtheta = vtheta
        self.vphi = vphi


class Accelerations:
    def __init__(self, ax, ay, az, apsi, atheta, aphi):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.apsi = apsi
        self.atheta = atheta
        self.aphi = aphi


class Control:
    thrust1 = []  # in the direction of (1,1)
    thrust2 = []  # in the direction of (1,-1)
    thrust3 = []  # in the direction of (-1,-1)
    thrust4 = []  # in the direction of (-1,1)

    def __init__(self, t1, t2, t3, t4):
        self.thrust1.append(t1)
        self.thrust2.append(t2)
        self.thrust3.append(t3)
        self.thrust4.append(t4)


class Wind:
    x = 0
    y = 0
    z = 0

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def toList(self):
        return [self.x, self.y, self.z]


class BlowingWind:
    wind = Wind(0, 0, 0)
    speed = 0
    std = 0
    t = 0

    def __init__(self, speed, std):
        self.speed = speed
        self.std = std
        self.wind.x = speed

    def generate(self):
        self.t += 0.01
        f = 0.01
        self.wind.x += np.random.normal(0, self.std) - 1 / 4 * (self.wind.x - self.speed)
        self.wind.y += np.random.normal(0, self.std) - 1 / 4 * self.wind.y
        self.wind.z += np.random.normal(0, self.std) - 1 / 4 * self.wind.z
        return self.wind

    def toList(self):
        return [self.wind.x, self.wind.y, self.wind.z]


class SinusoidalWind:
    wind = Wind(0, 0, 0)
    t = 0

    def __init__(self, ax, ay, az, freq):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.phix = np.pi / 2
        self.phiy = 0.233
        self.phiz = 0.343
        self.freq = freq

    def toList(self):
        return [self.wind.x, self.wind.y, self.wind.z]

    def generate(self):
        self.t += 0.01
        self.wind.x = self.ax * np.sin(2 * np.pi * self.freq * self.t + self.phix)
        self.wind.y = self.ay * np.sin(2 * np.pi * self.freq * self.t + self.phiy)
        self.wind.z = self.ay * np.sin(2 * np.pi * self.freq * self.t + self.phiz)
        return self.wind


class Drone:
    psi = [0]  # yaw;   z
    theta = [0]  # pitch; y
    phi = [0]  # roll;  x
    x = [0]
    y = [0]
    z = [0]
    ax = 0
    ay = 0
    az = 0
    ms = 0  # mass of center sphere
    rs = 0  # radius of center sphere
    mr = 0  # mass of each rotor
    Ir = 0  # moment of inertia of each rotor
    deltaT = 0.05
    control = Control(0, 0, 0, 0)
    breakflag = 0

    def __init__(self, deltaT, ms, rs, mr, Ir, speed):
        self.deltaT = deltaT
        self.ms = ms
        self.rs = rs
        self.mr = mr
        self.Ir = Ir
        self.control = Control(0,0,0,0)
        self.breakflag = 0
        self.psi = [0]  # yaw;   z
        self.theta = [0]  # pitch; y
        self.phi = [0]  # roll;  x
        self.x = [0]
        self.y = [0]
        self.z = [0]
        self.ax = 0
        self.ay = 0
        self.az = 0
        # speed: x-component of the velocity of wind at t = 0
        temp_thrust = np.sqrt((self._wind2force(Wind(speed, 0, 0)))[0] ** 2 + (gravity[2] * mass) ** 2) / 4
        self.control.thrust1.append(temp_thrust)
        self.control.thrust2.append(temp_thrust)
        self.control.thrust3.append(temp_thrust)
        self.control.thrust4.append(temp_thrust)
        self.theta.append(- np.arctan(np.abs(self._wind2force(Wind(speed, 0, 0))[0]) \
                                      / (g * mass)))

    def _wind2force(self, wind):
        # drag coefficient
        cd = 0.5

        # density of air
        rho = 1.293

        # cross section
        A = np.pi * self.rs ** 2

        # velocity of CoM
        v = self.velocities()
        u = [v.vx, v.vy, v.vz]

        # relative velocity in ground frame
        ur = np.array(u) - np.array(wind.toList())

        # relative speed
        sr = np.linalg.norm(ur)

        # air drag in ground frame
        F = - 1 / 2 * rho * A * sr * ur * 0.5

        return F  # a function which gives force considering wind speed

    def _updateControl(self):
        distance = np.linalg.norm([self.x[-1], self.y[-1], self.z[-1]])
        if (distance >= 0.2):
           self.breakflag = 1

        qualifier = max(distance, 0.13)
        thrust1 = self.control.thrust1[-1]
        thrust2 = self.control.thrust2[-1]
        thrust3 = self.control.thrust3[-1]
        thrust4 = self.control.thrust4[-1]

        p1 = 80 * qualifier
        p2 = 90 * qualifier
        p3 = 80 * qualifier

        thrust1 -= np.sign(self.sigma) * p1
        thrust2 -= np.sign(self.sigma) * p1
        thrust3 -= np.sign(self.sigma) * p1
        thrust4 -= np.sign(self.sigma) * p1

        thrust4 -= np.sign(self.w1) * p2
        thrust2 += np.sign(self.w1) * p2
        thrust3 -= np.sign(self.w2) * p3
        thrust1 += np.sign(self.w2) * p3

        thrust1 = _thresholdControl(thrust1)
        thrust2 = _thresholdControl(thrust2)
        thrust3 = _thresholdControl(thrust3)
        thrust4 = _thresholdControl(thrust4)

        self.control.thrust1.append(thrust1)
        self.control.thrust2.append(thrust2)
        self.control.thrust3.append(thrust3)
        self.control.thrust4.append(thrust4)

    #  def update(self,wind): # The wind is supposed to be blown in the x direction

    def velo(self, list):
        return (list[-1] - list[-2]) / self.deltaT

    def acc(self, list):
        if len(list) >= 3:
            return (list[-1] - 2 * list[-2] + list[-3]) / self.deltaT ** 2
        else:
            return 0

    def velocities(self):
        if len(self.x) == 1:
            return Velocities(0, 0, 0, 0, 0, 0)
        else:
            return Velocities(self.velo(self.x), self.velo(self.y), self.velo(self.z), self.velo(self.psi),
                              self.velo(self.theta), self.velo(self.phi))

    def accelerations(self):
        if len(self.x.len) <= 2:
            return Accelerations(0, 0, 0, 0, 0, 0)
        else:
            return Accelerations(self.acc(self.x), self.acc(self.y), self.acc(self.z), self.acc(self.psi),
                                 self.acc(self.theta), self.acc(self.phi))

    def update(self, wind):
        # base transition matrix
        rot = self.rotationMatrix()

        # Force caused by wind in ground frame
        fWind = self._wind2force(wind)

        # wind force in drone frame
        wind_drone = np.matmul(np.transpose(rot), fWind)

        # gravitational acceleration in drone frame
        g_drone = np.matmul(np.transpose(rot), gravity)

        # current thrust forces in ground frame
        f1 = self.control.thrust1[-1]
        f2 = self.control.thrust2[-1]
        f3 = self.control.thrust3[-1]
        f4 = self.control.thrust4[-1]

        # total upward force in the drone frame
        T = f1 + f2 + f3 + f4

        # total upward force vector in the drone frame
        T_drone = np.transpose([0, 0, T])

        # get the first-order derivative of the coordinates in the ground frame
        v = self.velocities()

        # translational acceleration in ground frame by Newton II
        translationalAcceleration_ground = gravity + (1 / mass) * np.matmul(rot, T_drone) + (1 / mass) * fWind

        # extract angular derivatives of Euler angles and convert to drone frame
        angularVelo_ground = [v.vphi, v.vtheta, v.vpsi]
        translationalVelo_ground = [v.vx, v.vy, v.vz]
        omega = np.matmul(self.angularVeloToDrone(), angularVelo_ground)

        # torques provided by the rotors in the drone frame
        tau = [d * (f4 - f2), d * (f3 - f1), 0]

        # from the thrust forces, obtain the angular velocities of the rotors in the drone frame
        rotorOmega1 = self.rotorAngularSpeed(f1)
        rotorOmega2 = self.rotorAngularSpeed(f2)
        rotorOmega3 = self.rotorAngularSpeed(f3)
        rotorOmega4 = self.rotorAngularSpeed(f4)
        # the angular velocity term in gyroscope torque
        omegaTau = rotorOmega1 - rotorOmega2 + rotorOmega3 - rotorOmega4

        # angular acceleration in drone frame by Euler-Newton equation
        angularAcceleration_drone = np.matmul(np.linalg.inv(self.sphereInertia()), - np.cross(omega, np.matmul(
            self.sphereInertia(), omega)) - self.Ir * np.cross(omega, [0, 0, 1]) * omegaTau + tau)

        # update the vx, vy, vz, x, y, z according to the derivatives obtained above
        translationalVelo_ground_new = self.deltaT * translationalAcceleration_ground + translationalVelo_ground
        self.x.append(self.x[-1] + self.deltaT * translationalVelo_ground_new[0])
        self.y.append(self.y[-1] + self.deltaT * translationalVelo_ground_new[1])
        self.z.append(self.z[-1] + self.deltaT * translationalVelo_ground_new[2])

        # update the Euler angles
        # first get the new angular velocities in the drone frame
        omega_new = omega + self.deltaT * angularAcceleration_drone

        # switch to ground frame
        angularVelo_ground_new = np.matmul(np.linalg.inv(self.angularVeloToDrone()), omega_new)

        self.phi.append(self.phi[-1] + self.deltaT * angularVelo_ground_new[0])
        self.theta.append(self.theta[-1] + self.deltaT * angularVelo_ground_new[1])
        self.psi.append(self.psi[-1] + self.deltaT * angularVelo_ground_new[2])

        # update acceleration
        self.ax = self.acc(self.x)
        self.ay = self.acc(self.y)
        self.az = self.acc(self.z)

        # scalars for controlling
        self.sigma = rot[0][2] * self.x[-1] + rot[1][2] * self.y[-1] + rot[2][2] * self.z[-1]
        wi = np.matmul(np.linalg.inv(self.angularVeloToDrone()), np.linalg.inv(self.sphereInertia()))
        self.w1 = np.inner(angularVelo_ground_new, wi[:, 0])
        self.w2 = np.inner(angularVelo_ground_new, wi[:, 1])

        # update control signal
        self._updateControl()


    # base transition matrix between the drone frame and the ground frame
    def rotationMatrix(self):
        cps = np.cos(self.psi[-1])
        sps = np.sin(self.psi[-1])
        cph = np.cos(self.phi[-1])
        sph = np.sin(self.phi[-1])
        ct = np.cos(self.theta[-1])
        st = np.sin(self.theta[-1])
        mat = [[cps * ct, cps * st * sph - sps * cph, cps * st * cph + sps * sph],
               [sps * ct, sps * st * sph + cps * cph, sps * st * cph - cps * sph],
               [-st, ct * sph, ct * cph]]
        return mat

    # transformation for angular velocity
    # result * (angular velocity in ground frame)
    # = (angular velocity in drone frame)
    # W_{\eta}
    def angularVeloToDrone(self):
        cps = np.cos(self.psi[-1])
        sps = np.sin(self.psi[-1])
        cph = np.cos(self.phi[-1])
        sph = np.sin(self.phi[-1])
        ct = np.cos(self.theta[-1])
        st = np.sin(self.theta[-1])
        mat = [[1, 0, -st],
               [0, cph, -sph],
               [0, -sph, ct * cph]]
        return mat

    def derivativeOfAngularVeloDroneToGround(self, omega):
        vphi = omega[0]
        vtheta = omega[1]
        vpsi = omega[2]
        cps = np.cos(self.psi[-1])
        sps = np.sin(self.psi[-1])
        cph = np.cos(self.phi[-1])
        sph = np.sin(self.phi[-1])
        ct = np.cos(self.theta[-1])
        st = np.sin(self.theta[-1])
        tt = np.tan(self.theta[-1])
        mat = [[0, vphi * cph * tt + vtheta * sph / ct ** 2, - vphi * sph * ct + vtheta * cph / ct ** 2],
               [0, - vphi * sph, - vphi * cph],
               [0, vphi * cph / ct + vphi * sph * tt / ct, - vphi * sph / ct + vtheta * cph * tt / ct]]

    def sphereInertia(self):
        Ixx = 1 / 5 * self.ms * self.mr ** 2
        return [[Ixx, 0, 0], [0, Ixx, 0], [0, 0, Ixx]]

    def rotorAngularSpeed(self, thrust):
        return np.sqrt(np.abs(thrust / rotorAngularSpeedConstant))

