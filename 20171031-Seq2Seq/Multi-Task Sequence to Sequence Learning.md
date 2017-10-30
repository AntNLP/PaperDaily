# <center>Multi-Task Sequence to Sequence Learning
<center>Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, 

Oriol Vinyals, Lukasz Kaiser
Google Brain

---
# One-To-Many Setting
![](figure1.PNG)
Only encoder shared.

---
# Many-To-One Setting
![](figure2.PNG)
Only decoder shared.

---
# Many-To-Many Setting
![](figure3.PNG)
Both encoder and decoder shared.

---

## Auto-Encoder
X -> z -> X

## Skip-Thoughts
Xt -> z -> Xt-1

---
# Experiments
## Large Tasks with Small Tasks
![](ex1.PNG)
many-to-one
Main task (Translation)
Auxiliary task (Paring-PTB)
**Help each other, but the mixing ratio is very important**

---
## Large Tasks with Medium Tasks
![](ex2.PNG)
one-to-many
Main task (Translation)
Auxiliary task (image caption)
**Help each other, but the mixing ratio is very important**

---
## Large Tasks with Large Tasks
![](ex3.PNG)
many-to-one
Main task (Translation)
Auxiliary task (Parsing-HC)
**Help each other, but the mixing ratio is very important**

---
## Large-Corpus Parsing Experiment
![](ex4.PNG)
many-to-one
Auto-encoder and Translation help HC Parsing less.

---
## Multi-Tasks and Unsupervised Learning
![](ex5.PNG)
many-to-many
auto-encoder and skip-thought is benefit for translation task, and the mixing ratio is very important.

---
## Conclusion
Multi-related-tasks learnt jointly can help each other, but the mixing learning ratio is very important.



