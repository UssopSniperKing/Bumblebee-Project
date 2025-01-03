from data import KinematicsSolutionHolder
import matplotlib.pyplot as plt
from core import Referential

def plot_kinematics(Holder: KinematicsSolutionHolder) -> None:
   
    plot_angles(Holder)
    plot_u_tip(Holder)
    plot_omega(Holder)
    plot_e_unit(Holder)
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

def plot_u_tip(Holder) -> None:

    Holder.u_tip.set_referential(Referential.GLOBAL)    
    print(Holder.u_tip.referential)
    plt.figure()
    plt.plot(Holder.time, Holder.u_tip.coords[0,:], label='u_g_x')
    plt.plot(Holder.time, Holder.u_tip.coords[1,:], label='u_g_y')
    plt.plot(Holder.time, Holder.u_tip.coords[2,:], label='u_g_z')
    plt.legend()


def plot_omega(Holder) -> None:

    Holder.omega.set_referential(Referential.GLOBAL)    
    plt.figure()
    plt.plot(Holder.time, Holder.omega.coords[0,:], label='omega_g_x')
    plt.plot(Holder.time, Holder.omega.coords[1,:], label='omega_g_y')
    plt.plot(Holder.time, Holder.omega.coords[2,:], label='omega_g_z')
    plt.legend()
    plt.title('omega global')

def plot_e_unit(Holder) -> None:

    print(Holder.ez)
    Holder.ez.set_referential(Referential.WING)
    print(Holder.ez)
    Holder.ez.set_referential(Referential.GLOBAL)
    print(Holder.ez)

    plt.figure()
    plt.plot(Holder.time, Holder.ez.coords[0,:], label='ez_w_g_x')
    plt.plot(Holder.time, Holder.ez.coords[1,:], label='ez_w_g_y')
    plt.plot(Holder.time, Holder.ez.coords[2,:], label='ez_w_g_z')
    plt.legend()
    plt.title('ez global')



#def plot_forces():