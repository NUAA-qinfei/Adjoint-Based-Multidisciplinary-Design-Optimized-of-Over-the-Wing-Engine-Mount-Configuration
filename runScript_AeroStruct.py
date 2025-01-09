#!/usr/bin/env python
import os
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from pyspline import Curve
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from tacs.mphys import TacsBuilder
from funtofem.mphys import MeldBuilder
from mphys.scenario_aerostructural import ScenarioAeroStructural
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils
import tacsSetup

parser = argparse.ArgumentParser()
# which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
# which task to run. Options are: opt (default), runPrimal, runAdjoint, checkTotals
parser.add_argument("-task", help="type of run to do", type=str, default="opt")
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================
#LScale = 1.0  # scale such that the L0=1

M = 0.77
T0 = 229.733
c = float(np.sqrt(1.4 * 287 * T0))
U0 = c * M
p0 = 30800.7
rho0 = p0 / T0 / 287.0
nuTilda0 = 3.196e-5
CL_target = 0.37
A0 = 47.9
alpha0 = 2.75
def calcUAndDir(U0, alpha0):
    dragDir = [float(np.cos(alpha0 * np.pi / 180)),float(np.sin(alpha0 * np.pi / 180)),0.0]
    liftDir = [float(-np.sin(alpha0 * np.pi / 180)),float(np.cos(alpha0 * np.pi / 180)),0.0]
    inletU =  [float(U0 * np.cos(alpha0 * np.pi / 180)),float(U0 * np.sin(alpha0 * np.pi / 180)),0.0]
    return inletU, dragDir, liftDir

inletU, dragDir, liftDir = calcUAndDir(U0, alpha0)
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 2e-5,
    "primalMinResTolDiff": 1e3,
    "couplingInfo": {
        "aerostructural": {"active": True, "pRef": p0, "propMovement": False, "couplingSurfaceGroups": {"wingGroup": ["wing"]}}
    },  # set the ref pressure for computing force for FSI
    "primalBC": {
        "U0": {"variable": "U", "patches": ["far"], "value": inletU},
        "p0": {"variable": "p", "patches": ["far"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["far"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["far"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "objFunc": {
        "CD": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "fixedDirection",
                "direction": dragDir,
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
        "CL": {
            "part1": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "fixedDirection",
                "direction": liftDir,
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                "addToAdjoint": True,
            }
        },
    },
    "adjEqnOption": {
        "gmresRelTol": 1.0e-2,  # set relative tolerance for block Gauss-Seidel adjoint
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
        "useNonZeroInitGuess": True,
    },
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "checkMeshThreshold": {
        "maxAspectRatio": 8000.0,
        "maxNonOrth": 90.0,
        "maxSkewness": 25.0,
    },
    "transonicPCOption": 2,
    "designVar": {
        #"aoa": {"designVarType": "AOA", "patches": ["far"], "flowAxis": "x", "normalAxis": "y"},
        "twist": {"designVarType": "FFD"},
        "shape": {"designVarType": "FFD"},
        "translatenacelle": {"designVarType": "FFD"},
    },
}

# Mesh deformation setup
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
}

# TACS Setup
tacsOptions = {
    "element_callback": tacsSetup.element_callback,
    "problem_setup": tacsSetup.problem_setup,
    "mesh_file": "./wingbox.bdf",
}


class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        aero_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerostructural")
        aero_builder.initialize(self.comm)

        # add the aerodynamic mesh component
        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        # create the builder to initialize TACS
        struct_builder = TacsBuilder(tacsOptions)
        struct_builder.initialize(self.comm)

        # add the structure mesh component
        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        # load and displacement transfer builder (meld), isym sets the symmetry plan axis (k)
        xfer_builder = MeldBuilder(aero_builder, struct_builder, isym=2, check_partials=True)
        xfer_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/ALLFFD.xyz", type="ffd"))

        # primal and adjoint solution options, i.e., nonlinear block Gauss-Seidel for aerostructural analysis
        # and linear block Gauss-Seidel for the coupled adjoint
        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-8)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-6, atol=1e-6)
        # add the coupling aerostructural scenario
        self.mphys_add_scenario(
            "cruise",
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=xfer_builder
            ),
            nonlinear_solver,
            linear_solver,
        )

        # need to manually connect the vars in the geo component to cruise
        for discipline in ["aero"]:
            self.connect("geometry.x_%s0" % discipline, "cruise.x_%s0_masked" % discipline)
        for discipline in ["struct"]:
            self.connect("geometry.x_%s0" % discipline, "cruise.x_%s0" % discipline)

        # add the structural thickness DVs
        ndv_struct = struct_builder.get_ndv()
        dvs.add_output("dv_struct", np.array(ndv_struct * [0.01]))
        self.connect("dv_struct", "cruise.dv_struct")

        # more manual connection
        self.connect("mesh_aero.x_aero0", "geometry.x_aero_in")
        self.connect("mesh_struct.x_struct0", "geometry.x_struct_in")

    def configure(self):

        # call this to configure the coupling solver
        super().configure()

        # add the objective function to the cruise scenario
        self.cruise.aero_post.mphys_add_funcs()

        # get the surface coordinates from the mesh component
        points = self.mesh_aero.mphys_get_surface_mesh()

        # add pointset for both aero and struct
        self.geometry.nom_add_discipline_coords("aero", points)
        self.geometry.nom_add_discipline_coords("struct")

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh_aero.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # Create reference axis for the twist variable
        nRefAxPts = self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k",volumes=[0])

        # Set up global design variables. We dont change the root twist
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wingAxis"].coef[i] = -val[i - 1]

        
        # Create reference axis for the nacell twist variables
        # Note here we set raySize=5 to avoid the warning when having highly skewed FFDs
        # "ray might not have been longenough to intersect the nearest curve."
        xFlap = [6.413, 6.413]
        yFlap = [1.472, 1.472]
        zFlap = [3.519, 5.287]
        cFlap = Curve(x=xFlap, y=yFlap, z=zFlap, k=2)
        self.geometry.nom_addRefAxis(name="nacelleAxis", curve=cFlap, axis="z", volumes=[1], raySize=5)

        # def nacelletwist(val, geo):
            # for i in range(2):
                # geo.rot_z["flapAxis"].coef[i] = -val[0]

        def translatenacelle(val, geo):
            C = geo.extractCoef("nacelleAxis")
            dx = val[0]
            dy = val[1]
            dz = val[2]
            for i in range(len(C)):
                C[i, 0] = C[i, 0] + dx
            for i in range(len(C)):
                C[i, 1] = C[i, 1] + dy
            for i in range(len(C)):
                C[i, 2] = C[i, 2] + dz
            geo.restoreCoef(C, "nacelleAxis")
            
        # add the global shape variable
        self.geometry.nom_addGlobalDV(dvName="translatenacelle", value=np.zeros(3), func=translatenacelle)
        
        # add twist variable
        self.geometry.nom_addGlobalDV(dvName="twist", value=np.array([0] * (nRefAxPts - 1)), func=twist)

        # add shape variable
        pts = self.geometry.DVGeo.getLocalIndex(0)
        indexList = pts[:, :, :].flatten()
        PS = geo_utils.PointSelect("list", indexList)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape", axis="y", pointSelect=PS)

        # setup the volume and thickness constraints
        #leRoot = np.array([0.1 * LScale,0, 0.01 * LScale])
        #leBreak = np.array([2.647 * LScale,0, 4.8 * LScale])
        #leTip = np.array([7.5 * LScale, 0,13.95 * LScale])
        #rootChord = 6 * LScale
        #breakChord = 3.8 * LScale
        #tipChord = 1.5* LScale
        #coe1 = 0  # in production run where the mesh is refined, set coe1=0.01
        #coe2 = 0.98
        #xaxis = np.array([1.0, 0, 0])
        #leList = [leRoot + coe1 * rootChord * xaxis, leBreak + coe1 * breakChord * xaxis, leTip + coe1 * tipChord * xaxis]
        #teList = [leRoot + coe2 * rootChord * xaxis, leBreak + coe2 * breakChord * xaxis, leTip + coe2 * tipChord * xaxis]
        leList = [[0.1, 0, 0.01], [2.647, 0, 4.8], [7.5, 0, 13.95]]
        teList = [[5.967, 0, 0.01], [6.333, 0, 4.8], [8.95, 0, 13.95]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=30, nChord=15)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=30, nChord=15)
        
        # add the LE/TE constraints
        self.geometry.nom_add_LETEConstraint("lecon", volID=0, faceID="iLow")
        self.geometry.nom_add_LETEConstraint("tecon", volID=0, faceID="iHigh")

        # add the design variables to the dvs component's output
        self.dvs.add_output("twist", val=np.array([0] * (nRefAxPts - 1)))
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        #self.dvs.add_output("aoa", val=np.array([aoa0]))
        self.dvs.add_output("translatenacelle", val=np.zeros(3))
        # manually connect the dvs output to the geometry and cruise
        self.connect("twist", "geometry.twist")
        self.connect("shape", "geometry.shape")
        self.connect("translatenacelle", "geometry.translatenacelle")
        #self.connect("aoa", "cruise.aoa")

        # define the design variables
        self.add_design_var("twist", lower=-0.5, upper=0.5, scaler=1.0)
        self.add_design_var("shape", lower=-0.015, upper=0.015, scaler=1.0)
        #self.add_design_var("aoa", lower=2.74, upper=2.76, scaler=1.0)
        self.add_design_var("translatenacelle", lower=[-1,-0.3,-1], upper=[1,1,1], scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=CL_target, scaler=1.0)
        # stress constraint
        self.add_constraint("cruise.ks_vmfailure", lower=0.0, upper=2.0, scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        self.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero_struct.html")

# initialize the optimization function
optFuncs = OptFuncs(daOptions, prob)

# use pyoptsparse to setup optimization
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer
# options for optimizers
if args.optimizer == "SNOPT":
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-5,
        "Major optimality tolerance": 1.0e-5,
        "Minor feasibility tolerance": 1.0e-5,
        "Verify level": -1,
        "Function precision": 1.0e-5,
        "Major iterations limit": 100,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol": 1.0e-5,
        "constr_viol_tol": 1.0e-5,
        "max_iter": 100,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.optimizer == "SLSQP":
    prob.driver.opt_settings = {
        "ACC": 1.0e-5,
        "MAXIT": 100,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("optimizer arg not valid!")
    exit(1)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"

if args.task == "opt":
    # solve CL
    #optFuncs.findFeasibleDesign(["cruise.aero_post.CL"], ["aoa"], targets=[CL_target])
    # run the optimization
    prob.run_driver()
elif args.task == "runPrimal":
    # just run the primal once
    prob.run_model()
elif args.task == "runAdjoint":
    # just run the primal and adjoint once
    prob.run_model()
    totals = prob.compute_totals()
    if MPI.COMM_WORLD.rank == 0:
        print(totals)
elif args.task == "checkTotals":
    # verify the total derivatives against the finite-difference
    prob.run_model()
    prob.check_totals(
        of=["cruise.aero_post.CD", "cruise.aero_post.CL"],
        wrt=["shape", "aoa"],
        compact_print=True,
        step=1e-3,
        form="central",
        step_calc="abs",
    )
else:
    print("task arg not found!")
    exit(1)
