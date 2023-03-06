from klampt.plan.cspace import CSpace,MotionPlan
import time
import numpy as np
import habitat_sim

class HabitatObstacleCSpace(CSpace):
    def __init__(self, sim: "HabitatSim", height):
        CSpace.__init__(self)
        #set bounds
        self._sim = sim
        self.height = height
        bounds = self._sim.pathfinder.get_bounds()
        self.bound = [(bounds[0][0], bounds[1][0]), (bounds[0][2], bounds[1][2])]
        #set collision checking resolution
        self.eps = 1e-3

    def feasible(self, q):
        #bounds test
        # print('the current q is', q)
        if not self._sim.pathfinder.is_navigable(np.asarray([q[0], self.height, q[1]])):
            return False
        # print('this point is navigable')
        # a_x, a_y = maps.to_grid(
        #     agent_position[2],
        #     agent_position[0],
        #     self._top_down_map.shape[0:2],
        #     sim=self._sim,
        # )

        return True

    def sample(self):
        """Overload this to define a nonuniform sampler.
        By default, it will sample from the axis-aligned bounding box
        defined by self.bound. To define a different domain, set self.bound
        to the desired bound.
        """
        p = self._sim.pathfinder.get_random_navigable_point()
        return [p[0], p[2]]

    def sampleneighborhood(self, c, r):
        """Overload this to define a nonuniform sampler.
        By default, it will sample from the axis-aligned box of radius r
        around c, but clamped to the bound.
        """
        p = self._sim.pathfinder.get_random_navigable_point_near(np.asarray([c[0], self.height, c[1]]), r)
        return [p[0], p[2]]

class CSpaceObstacleProgram():
    def __init__(self, space, start=(0.1,0.5), goal=(0.9,0.5)):
        self.space = space
        #PRM planner
        MotionPlan.setOptions(type="prm", knn=10, connectionThreshold=0.1)
        self.optimizingPlanner = False

        #FMM* planner
        #MotionPlan.setOptions(type="fmm*")
        #self.optimizingPlanner = True

        #RRT planner
        #MotionPlan.setOptions(type="rrt",perturbationRadius=0.25,bidirectional=True)
        #self.optimizingPlanner = False

        #RRT* planner
        #MotionPlan.setOptions(type="rrt*")
        #self.optimizingPlanner = True

        #random-restart RRT planner
        #MotionPlan.setOptions(type="rrt",perturbationRadius=0.25,bidirectional=True,shortcut=True,restart=True,restartTermCond="{foundSolution:1,maxIters:1000}")
        #self.optimizingPlanner = True

        #OMPL planners:
        #Tested to work fine with OMPL's prm, lazyprm, prm*, lazyprm*, rrt, rrt*, rrtconnect, lazyrrt, lbtrrt, sbl, bitstar.
        #Note that lbtrrt doesn't seem to continue after first iteration.
        #Note that stride, pdst, and fmt do not work properly...
        #MotionPlan.setOptions(type="ompl:rrt",suboptimalityFactor=0.1,knn=10,connectionThreshold=0.1)
        #self.optimizingPlanner = True

        self.planner = MotionPlan(space)
        self.start = (start[0], start[2])
        self.goal = (goal[0], goal[2])
        self.planner.setEndpoints(self.start, self.goal)
        self.points = []

        if self.space.feasible(self.start):
            print('start is feasible')
        else:
            print('start is not feasible')

        if self.space.feasible(self.goal):
            print('goal is feasible')
        else:
            print('goal is not feasible')

    def run(self):
        t0 = time.time()
        #max 20 seconds of planning
        while (self.optimizingPlanner or not self.points) and time.time() - t0 < 20:
            self.planner.planMore(100)
            self.points = self.planner.getPath()
            self.G = self.planner.getRoadmap()

        if self.points:
            print("Solved, path has", len(self.points), "milestones")
            return True
        else:
            print("Not solved")
            return False

    def produce_path(self):
        path = habitat_sim.ShortestPath()
        points_3d = []
        for p in self.points:
            points_3d.append(np.asarray([p[0], self.space.height, p[1]]))
        path.points = points_3d
        path.geodesic_distance = self.planner.pathCost(self.points)
        print('the total length is ', path.geodesic_distance)
        return path

if __name__=='__main__':
    space = HabitatObstacleCSpace()

    start = (0.06, 0.5)
    goal = (0.94, 0.5)

    program = CSpaceObstacleProgram(space, start, goal)
    program.run()
    print(program.path)

