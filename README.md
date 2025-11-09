# mobile-robot

Simulation of a mobile robot using EKF-SLAM, occupancy grid mapping and natural evolution strategies (NES), built for the course "**Autonomous Robotic Systems**" (2025) in M.Sc. Artificial Intelligence at Maastricht University.

### Fitness and diversity of the evolved agent over time

<p align="center">
    <img src="figures/plot.png" alt="" width=1000/>
</p>

1D example of an optimization process using natural evolution strategies. The solution is $x=9$ (value with maximum fitness) and the initialization is at $x=0$. Over 20 generations, the parameters of the search distribution are updated such that its mean is approximately at the solution value. 

<p align="center">
    <img src="figures/nes2.gif?v=1" alt="" style="width: 970px; height: auto;"/>
</p>

---

### Demonstration of the simulated robot's occupancy grid mapping and behavior

<p align="center">
    <img src="figures/occupancy2.gif?v=1" alt="" style="width: 500px; height: auto;"/>
</p>

<p align="center">
    <img src="figures/behavior1.gif?v=1" alt="" style="width: 500px; height: auto;"/>
</p>

Earlier version of the evolved policy under a different fitness function, which rewarded displacement from the spawn point.

<p align="center">
    <img src="figures/behavior2.gif?v=1" alt="" style="width: 500px; height: auto;"/>
</p>
