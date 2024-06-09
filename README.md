---
title: OR TermProject
date: 2024-06-08 00:00:39
author: Immortal-Fates , PhiFan , Twinkle ,  daichengjiang
---

# Main Takeaway



在运筹学课程上，我们小组解决图书馆预约书籍问题

提出了三个子问题

![工作结构](/presentation/structure.png)



**对于问题一**（TSP），我们采用贪心算法来解决智能机器人在图书馆进行派送任务时的路径规划问题。目标是规划其运动路径，使其行驶路径最短。通过建立相应的数学模型，并考虑每个途径点只能访问一次等约束条件，我们直接利用贪心算法进行求解，得出最优路径，提升了AGV的取书效率。

**对于问题二**，针对多AGV智能机器人同时取书的场景，我们将此问题建模为SDVRP（Split Delivery Vehicle Routing Problem）问题。通过贪婪算法和蚁群算法的结合，解决了多辆AGV的协同调度问题，确保每个机器人在载满书籍或未取满情况下合理返回传送处，最终求解出总行驶路径最短的调度方案。

**对于问题三**，提出了一种基于混合整数规划（MILP）的方法，用于解决仓储环境中托盘与书籍的高效拣选和调度问题。我们建立了一个详细的数学模型，考虑了托盘内书本数量与预约订单匹配、托盘与书籍挑选节点的匹配、托盘放回空储位节点以及空托盘回收等多个约束条件。通过定义变量和约束条件，模型确保了从托盘中挑选出的书籍数量满足预约订单需求，同时最小化了托盘在仓库中的移动距离。模型的主要约束包括：托盘内书本数量与预约订单匹配约束、托盘与书籍挑选节点的匹配约束、托盘与回收处和储位的约束。

<!--more-->



# Folder Tree

```
project/
│
├── Problem1/
│ ├── src 
│ ├── code 
│ └── README.md
│
├── Problem2/
│ ├── code
│ └── result
| └── src
| └── README.md
│
├── Problem3/
│ ├── solve
│ └── draw_map
| └── README.md
│
├── presentation/
│ ├── OR.pptx
│ └── OR.pdf
│
├── thesis/
│
└── README.md
```
