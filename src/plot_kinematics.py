from data import KinematicsSolutionHolder
import matplotlib.pyplot as plt
from core import Referential
from pathlib import Path

def plot_kinematics(Holder: KinematicsSolutionHolder, save_fig: bool, show_fig: bool) -> None:
   
    plot_angles(Holder)
    plot_u_tip(Holder)
    plot_omega(Holder)
    plot_e_unit(Holder)
    plot_angle_of_attack(Holder)
    plot_coefficients(Holder)
    plot_aero_vectors(Holder)
    plot_u_tip_dt(Holder)
    plot_omega_dt(Holder)
    plot_forces(Holder)
    plot_omega_planar_square(Holder)

    if save_fig:
        save_figures()
    
    if show_fig:
        plt.show()


def save_figures() -> None:

    figures_path = Path(__file__).parent.parent / "figures"
    figures_path.mkdir(exist_ok=True)

    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(figures_path / f"Figure_{i}.png")


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

def plot_u_tip_dt(Holder) -> None:

    Holder.u_tip_dt.set_referential(Referential.WING)    
    plt.figure()
    plt.plot(Holder.time, Holder.u_tip_dt.coords[0,:], label='u_dt_w_x')
    plt.plot(Holder.time, Holder.u_tip_dt.coords[1,:], label='u_dt_w_y')
    plt.plot(Holder.time, Holder.u_tip_dt.coords[2,:], label='u_dt_w_z')
    plt.legend()

def plot_omega_dt(Holder) -> None:

    Holder.omega_dt.set_referential(Referential.WING)    
    plt.figure()
    plt.plot(Holder.time, Holder.omega_dt.coords[0,:], label='omega_dt_w_x')
    plt.plot(Holder.time, Holder.omega_dt.coords[1,:], label='omega_dt_w_y')
    plt.plot(Holder.time, Holder.omega_dt.coords[2,:], label='omega_dt_w_z')
    plt.legend()
    plt.title('omega_dt wing')

def plot_forces(Holder):

    plt.figure()
    plt.plot(Holder.time, Holder.force_TC.coords[2,:], label='F_TC', color='yellow')
    plt.plot(Holder.time, Holder.force_TD.coords[2,:], label='F_TD', color='lime')
    plt.plot(Holder.time, Holder.force_RC.coords[2,:], label='F_RC', color='orange')
    plt.plot(Holder.time, Holder.force_AMz.coords[2,:], label='F_AMz', color='red')
    plt.plot(Holder.time, Holder.force_AMx.coords[2,:], label='F_AMx', color='red', linestyle='dashed')
    plt.plot(Holder.time, Holder.force_RD.coords[2,:], label='F_RD', color='green')
    plt.plot(Holder.time, Holder.force_QSM.coords[2,:], label='F_QSM', color='blue', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.title('Vertical components of forces in global coordinate system')
    plt.legend()

def plot_omega_planar_square(Holder: KinematicsSolutionHolder):

    Holder.omega_planar.set_referential(Referential.GLOBAL)

    plt.figure()
    plt.plot(Holder.time, Holder.omega_planar.norm() ** 2)

    plt.figure()
    plt.plot(Holder.time, Holder.lift_coeff)
    plt.plot(Holder.time, Holder.drag_coeff)