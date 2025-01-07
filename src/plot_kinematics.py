from data import KinematicsSolutionHolder
import matplotlib.pyplot as plt
from core import Referential

def plot_kinematics(Holder: KinematicsSolutionHolder) -> None:
   
    plot_angles(Holder)
    plot_u_tip(Holder)
    plot_omega(Holder)
    plot_e_unit(Holder)
    plot_angle_of_attack(Holder)
    plot_coefficients(Holder)
    plot_aero_vectors(Holder)
    plt.show()


def plot_angles(Holder: KinematicsSolutionHolder) -> None:

    plt.figure()
    plt.plot(Holder.time, Holder.phi.degrees, label='phi')
    plt.plot(Holder.time, Holder.alpha.degrees, label='alpha')
    plt.plot(Holder.time, Holder.theta.degrees, label='theta')
    plt.xlabel('time')
    plt.ylabel('angles (degrees)')
    plt.title('Angle')
    plt.legend()

    plt.figure()
    plt.plot(Holder.time, Holder.phi_dt.degrees, label='phi_dt')
    plt.plot(Holder.time, Holder.alpha_dt.degrees, label='alpha_dt')
    plt.plot(Holder.time, Holder.theta_dt.degrees, label='theta_dt')
    plt.xlabel('time')
    plt.ylabel('angles (degrees/s)')
    plt.title('Angles derivatives')
    plt.legend()

def plot_angle_of_attack(Holder):

    plt.figure()
    plt.plot(Holder.time, Holder.angle_of_attack.degrees, label='AoA')
    plt.xlabel('Time')
    plt.ylabel('Angle og Attack')
    plt.title('Angle of attack')
    plt.legend()


def plot_u_tip(Holder) -> None:

    Holder.u_tip.set_referential(Referential.GLOBAL)    
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

    Holder.ez.set_referential(Referential.GLOBAL)
    plt.figure()
    plt.plot(Holder.time, Holder.ez.coords[0,:], label='ez_w_g_x')
    plt.plot(Holder.time, Holder.ez.coords[1,:], label='ez_w_g_y')
    plt.plot(Holder.time, Holder.ez.coords[2,:], label='ez_w_g_z')
    plt.legend()
    plt.title('ez global')


def plot_coefficients(Holder):
    
    plt.figure()
    plt.plot(Holder.angle_of_attack.degrees, Holder.lift_coeff.T, label='CL')
    plt.plot(Holder.angle_of_attack.degrees, Holder.drag_coeff, label='CD')
    plt.xlabel('Angle of attack')
    plt.title('Aerodynamics coefficients')
    plt.legend()
   
def plot_aero_vectors(Holder):

    Holder.e_lift.set_referential(Referential.GLOBAL)
    Holder.e_drag.set_referential(Referential.GLOBAL)
    plt.figure()
    plt.plot(Holder.time, Holder.e_lift.coords[0,:], label='e_lift_g_x')
    plt.plot(Holder.time, Holder.e_lift.coords[1,:], label='e_lift_g_y')
    plt.plot(Holder.time, Holder.e_lift.coords[2,:], label='e_lift_g_z')
    plt.xlabel('time')
    plt.title('e_lift_g components')
    plt.legend()


    plt.figure()
    plt.plot(Holder.time, Holder.e_drag.coords[0,:], label='e_drag_g_x')
    plt.plot(Holder.time, Holder.e_drag.coords[1,:], label='e_drag_g_y')
    plt.plot(Holder.time, Holder.e_drag.coords[2,:], label='e_drag_g_z')
    plt.xlabel('time')
    plt.title('e_drag_g components')
    plt.legend()


#def plot_forces():