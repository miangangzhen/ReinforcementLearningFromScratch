"""
第三章MDPs笔记

强化学习是一系列的决策。包括对回馈的估计，在不同环境下选择不同的action，决策取决于state。
agent负责学习并做出决策。agent所交互的对象，叫做environment。
在持续不断的交互中，agent选择actions(At)，而env回应新的state(St)，并给出rewards(Rt)，agent需要最大化长期rewards。
agent与env在离散的时间轴上不断交互: S0; A0; R1; S1; A1; R2; S2; A2; R3; ...
在有限马尔科夫决策过程中，S A R集合包含有限个数的元素。
在这样的条件下，随机变量Rt和St在state和action条件下，有离散的概率分布。
p(s',r|s, a) = Pr{St=s', Rt=r | St-1=s, At-1=a}

state的构成可能是基于对过去感觉的记忆，甚至完全是精神或主观的
总体来说，actions可以是任何我们希望机器学习的决策，而states可以是我们认为对决策有意义的任何信息。

agent与env的界限应该这样划分: agent能够绝对控制的信息属于agent，此外都属于env。
有些信息，即使对agent透明，但agent不能任意控制它们，那么它们依然属于env。
实际上，当states,actions,rewards确定后，agent与env的界限就划分开了。

MDP框架是对交互中的目标导向学习问题的大量抽象。
actions代表agent的选择
states代表agent做决定时所依赖的信息
rewards代表agent的目标


"""

















