# homework01----
该项目完整实现了对于数据集的筛选，模型的构建，以及最终结果的输出与测试
# 输出结果
Here is the provided information formatted in Markdown:

```markdown
### 数据集形状： (506, 14)
| CRIM   | ZN   | INDUS | CHAS | NOX   | RM    | AGE   | DIS   | RAD | TAX  | PTRATIO | B     | LSTAT | MEDV |
|--------|------|-------|------|-------|-------|-------|-------|-----|------|---------|-------|-------|------|
| 0.00632 | 18.0 | 2.31  | 0    | 0.538 | 6.575 | 65.2  | 4.0900 | 1   | 296  | 15.3    | 396.90 | 4.98  | 24.0 |
| 0.02731 | 0.0  | 7.07  | 0    | 0.469 | 6.421 | 78.9  | 4.9671 | 2   | 242  | 17.8    | 396.90 | 9.14  | 21.6 |
| 0.02729 | 0.0  | 7.07  | 0    | 0.469 | 7.185 | 61.1  | 4.9671 | 2   | 242  | 17.8    | 392.83 | 4.03  | 34.7 |
| 0.03237 | 0.0  | 2.18  | 0    | 0.458 | 6.998 | 45.8  | 6.0622 | 3   | 222  | 18.7    | 394.63 | 2.94  | 33.4 |
| 0.06905 | 0.0  | 2.18  | 0    | 0.458 | 7.147 | 54.2  | 6.0622 | 3   | 222  | 18.7    | 396.90 | 5.33  | 36.2 |

### 各特征与 MEDV 的相关系数：
```
| Feature | Correlation with MEDV |
|---------|-----------------------|
| MEDV    | 1.000000              |
| RM      | 0.695360              |
| ZN      | 0.360445              |
| B       | 0.333461              |
| DIS     | 0.249929              |
| CHAS    | 0.175260              |
| AGE     | -0.376955             |
| RAD     | -0.381626             |
| CRIM    | -0.388305             |
| NOX     | -0.427321             |
| TAX     | -0.468536             |
| INDUS   | -0.483725             |
| PTRATIO | -0.507787             |
| LSTAT   | -0.737663             |

### 选取的主要特征：
- `RM`
- `PTRATIO`
- `LSTAT`

### 特征缺失值数量： 0

### 目标缺失值数量： 0

### 模型结构：
```python
RegressionNN(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=1, bias=True)
  )
)
```

### 训练过程：
```
Epoch [20/200], Loss: 0.2206
Epoch [40/200], Loss: 0.1923
Epoch [60/200], Loss: 0.1811
Epoch [80/200], Loss: 0.1728
Epoch [100/200], Loss: 0.1652
Epoch [120/200], Loss: 0.1578
Epoch [140/200], Loss: 0.1511
Epoch [160/200], Loss: 0.1438
Epoch [180/200], Loss: 0.1379
Epoch [200/200], Loss: 0.1292
```
```

This markdown structure provides a clear and organized representation of the dataset, correlation coefficients, model details, and training progress.
