#set page(margin: (x: 45pt, y: 30pt))
#set text(size: 13pt, font: ("STFangsong"), lang: "zh")
#set heading(numbering: "1.a.")
#set par(justify: true, first-line-indent: 2em)
#let indent = h(2em)

#align(center)[#text(size: 24pt, [PossionSolver])]

= 单元项计算
$ k_("ij")^((e)) = - integral_(Omega_k) nabla N_i dot nabla N_j d Omega - integral_(Omega_k) N_i N_j (partial f) / (partial u) d Omega $

$ f_i^((e)) = integral_(Omega_k) nabla N_i nabla u d Omega + integral_(Omega_k) f N_i d Omega $

位置变量可以表示为
$ u approx sum_i N_i u_i $

我们可以将积分离散化使用求和代替积分，使用高斯积分对进行近似计算，将$Omega$转换为参考单元如
$ integral_(Omega_k) f d Omega approx sum_(q=1)^(n_g) omega_q f(epsilon_q,eta_q) $
其中$n_g$为高斯积分点个数，$omega_q$为高斯积分点权重，$(xi_q,eta_q)$积分点坐标。
//$nabla$算符可以使用为对两个方向的偏微分代替
$ k_("ij")^((e)) approx -sum_(q=1)^(n_q)omega_q nabla N_i (xi_q,eta_q) dot nabla n_j (xi_q,eta_q) - sum_(q=1)^(n_q) omega_q N_i (xi_q,eta_q)N_j (xi_q,eta_q)(partial f)/(partial u)(xi_q,eta_q) $
其中，$nabla N_i$和$nabla N_j$可以通过形函数导数计算得到。

$ f_i^((e)) approx sum_(q=q)^(n_g) omega_q nabla N_i (xi_q,eta_q) dot nabla u (xi_q,eta_q) + sum_(q=1)^(n_g) omega_q f(xi_q,eta_q) N_i (xi_q,eta_q) $
其中，$u(xi_q,eta_q)$为插值得到的单元内位移。

= 单元刚度矩阵和荷载向量组装
#set math.mat(delim: "[")
$ k^((e)) = mat(
    k_(11)^((e)), k_(12)^((e)), dots.h, k_(1n_e)^((e)) ;
    k_(21)^((e)), k_(22)^((e)), dots.h, k_(2n_e)^((e)) ;
    dots.v, dots.v ,dots.down,dots.v;
    k_(n_e 1)^((e)), k_(n_e 2)^((e)), dots.h, k_(n_e n_e)^((e)) ;
) , f^((e)) = mat(
    f^((e))_1;f_2^((e)); dots.v;f_(n_e)^((e)) 
) $

- 全局刚度矩阵$K$大小为$n_("global") times n_("global")$，其中$n_("global")$是整个系统的自由度总数。
- 全局荷载向量$F$是整个计算域的荷载向量，大小为$n_("global")times 1$。

每个单元上的自由度是一组局部自由度，与全局自由度通过自由度编号表（Element Freedom Table, EFT）相关联。

单元刚度矩阵$k^((e))$对全局刚度矩阵的贡献为:
$ K["dof"_i,"dof"_j]=K["dof"_i,"dof"_j] + k^((e))_("ij"), forall i,j in {1,2,dots,n_e} $
其中$"dof"_i$为全局自由度

伪代码为
```
for e in elements:
    for i in range(n_e):
        global_i = EFT[e][i]  # 从单元到全局自由度的映射
        for j in range(n_e):
            global_j = EFT[e][j]
            K[global_i, global_j] += k[e][i, j]
```

#indent 单元载荷向量 $f^((e))$ 对全局载荷向量 $F$ 的贡献为：
$ F["dof"_i] = F["dof"_i] + f^((e))_i, forall i in {1,2,dots,n_e} $

为代码为
```
for e in elements:
    for i in range(n_e):
        global_i = EFT[e][i]  # 从单元到全局自由度的映射
        F[global_i] += f[e][i]
```



= 狄利克雷边界
#indent 假设我们要求解的问题为
$ K u =F $
其中$K$是全局刚度矩阵，$u$是未知量，$F$是全局荷载向量，那么狄利克雷边界条件施加规定节点上i的$u_i$的值为：
$ u_i = accent(u, macron)_i $
其中$accent(u, macron)_i$是边界上的已知值。

首先找到边界节点的index，假如边界节点索引合集为$cal(D)$，例如：
$ cal(D)={i_1,i_2,i_3,dots,i_m} $

接着我们修改刚度矩阵$K$，对于边界节点$i in cal(D)$，将全局刚度矩阵第i行和第i列修改为：
$ K[i,:]=0, K[:,i]=0, K=[i,i]=1 $

确保系统方程在边界处是固定的。修改荷载向量，对于$i in cal(D)$，将全局荷载向量第i个分量设置为边界值。
$ F[i]=accent(u, macron)_i $

伪代码 ```
for i, value in zip(boundary_nodes, boundary_values):
 K[i, :] = 0
 K[:, i] = 0
 K[i, i] = 1
 F[i] = value
```

