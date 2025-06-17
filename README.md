# practice
---

# 基于图神经网络的电力系统约束优化调度模型

## 一、项目背景与意义

### 1.1 项目背景

随着电力系统规模的不断扩大以及新能源渗透率的提升，传统的发电调度面临日益复杂的约束和动态变化环境。多发电机组的最优调度问题（Unit Commitment, UC）因其具有强耦合、高维度和强非线性等特征，成为电力系统优化中的核心难题。

### 1.2 项目意义

本项目旨在融合**图神经网络（GNN）**与**深度学习约束优化技术**，构建一种新型的调度优化框架，为智能电网调度提供更高效、智能的解决方案。

* **实际意义**：

  * 提高发电调度效率与自动化水平；
  * 降低能源消耗与发电成本；
  * 提升系统运行的稳定性与安全性；
  * 支持高比例可再生能源接入。

* **学术意义**：

  * 推动深度学习在电力系统优化中的应用；
  * 探索图数据建模与约束优化的结合方法；
  * 支持跨学科研究创新（电力系统 + 图神经网络 + 优化理论）。

---

## 二、现有方法分类与项目定位

### 2.1 当前主流方法类别

目前用于电力系统调度的优化方法主要包括以下几类：

| 方法类别                  | 特点                 | 优劣势             |
| --------------------- | ------------------ | --------------- |
| 传统数学优化方法（如 MILP, QP）  | 精度高，解析性强           | 对于大规模系统计算复杂度高   |
| 启发式算法（如遗传算法、粒子群）      | 易于实现，适应性强          | 缺乏全局最优性保证，收敛慢   |
| 强化学习                  | 适合动态环境，支持在线学习      | 学习稳定性差，对高维状态不敏感 |
| **图神经网络 + 深度优化（本项目）** | 能建模复杂拓扑结构，适应系统不确定性 | 算法设计复杂，需大数据训练   |

### 2.2 本项目方法定位

本项目方法属于第四类：**图神经网络驱动的约束优化方法**，重点解决电力系统中**复杂变量依赖关系**与**约束映射问题**，具有可扩展性强、泛化能力好的特点。

---

## 三、方法与技术实现

### 3.1 模型整体架构流程图

```
flowchart TD
    A[电力系统图数据采集] --> B[图结构构建 (节点:机组, 边:耦合关系)]
    B --> C[图神经网络特征提取 GNN]
    C --> D[调度决策变量生成]
    D --> E[约束优化模块（投影梯度/拉格朗日优化）]
    E --> F[生成最优调度策略]
```

---

### 3.2 方法核心组成

#### （1）图结构建模

* **节点**：发电机组、变电站、负载等；
* **边**：节点之间的电力传输关系或地理/经济耦合关系；
* **特征**：每个节点包含状态变量（功率上下限、成本系数、运行状态等）。

#### （2）图神经网络（GNN）表示学习

使用 GAT（Graph Attention Network） 结构，对每个节点嵌入向量进行建模：

$$
h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \cdot W h_j
$$

其中：

* $h_i$ 表示节点 $i$ 的特征；
* $\alpha_{ij}$ 为注意力权重，计算方式为：

$$
\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i \Vert Wh_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(a^T[Wh_i \Vert Wh_k]))}
$$

#### （3）调度决策生成与优化

* 初步决策输出为节点的功率调度值 $p_i$；
* 引入拉格朗日对偶优化形式进行可行性约束投影：

$$
\min_{p} \quad C(p) + \sum_k \lambda_k g_k(p) \quad \text{s.t.} \quad h_j(p) \leq 0
$$

采用神经网络内嵌的优化模块（如 Differentiable Optimization Layer）进行训练。

---

### 3.3 核心代码片段示例（伪代码）

```python
class PowerDispatchGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.gat1 = GATConv(node_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)  # 输出功率

    def forward(self, x, edge_index):
        h = F.relu(self.gat1(x, edge_index))
        h = F.relu(self.gat2(h, edge_index))
        p = self.output_layer(h)  # 调度输出
        return p
```
![image](https://github.com/user-attachments/assets/94151391-e0b9-470c-a47e-a9450aa6599c)
![image](https://github.com/user-attachments/assets/ef990982-d8c4-4a93-8f29-54126bd91000)
---
