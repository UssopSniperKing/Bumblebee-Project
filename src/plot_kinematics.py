from data import KinematicsSolutionHolder
import matplotlib.pyplot as plt

def plot_kinematics(Holder: KinematicsSolutionHolder) -> None:
   
    plot_angles(Holder)
    plt.show()


def plot_angles(Holder: KinematicsSolutionHolder) -> None:

    plt.figure()
    plt.plot(Holder.time, Holder.phi.radians, label='phi')
    plt.plot(Holder.time, Holder.alpha.radians, label='alpha')
    plt.plot(Holder.time, Holder.theta.radians, label='theta')
    plt.xlabel('time')
    plt.ylabel('angles')
    plt.title('Angle')
    plt.legend()

    plt.figure()
    plt.plot(Holder.time, Holder.phi_dt.radians, label='phi_dt')
    plt.plot(Holder.time, Holder.alpha_dt.radians, label='alpha_dt')
    plt.plot(Holder.time, Holder.theta_dt.radians, label='theta_dt')
    plt.xlabel('time')
    plt.ylabel('angles')
    plt.title('Angles derivatives')
    plt.legend()





#def plot_forces():