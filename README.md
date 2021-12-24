# Curriculum Learning for Safe Mapless Navigation

The poster version of this paper has been accepted by the 37th ACM/SIGAPPSymposium on Applied Computing (SAC ’22), April 25–29, 2022, Virtual Event.
The full paper is available at this [link](https://arxiv.org/abs/2112.12490)

### Authors
* **Luca Marzari** - luca.marzari@studenti.univr.it
*  **Davide Corsi** - davide.corsi@univr.it
*  **Enrico Marchesini** - enrico.marchesini@univr.it
*  **Alessandro Farinelli** - alessandro.farinelli@univr.it

## Abstract
This work investigates the effects of Curriculum Learning (CL)-based approaches on the agent's performance. In particular, we focus on the safety aspect of robotic mapless navigation, comparing over a standard end-to-end (E2E) training strategy. To this end, we present a CL approach that leverages Transfer of Learning (ToL) and fine-tuning in a Unity-based simulation with the Robotnik Kairos as a robotic agent. For a fair comparison, our evaluation considers an equal computational demand for every learning approach (i.e., the same number of interactions and difficulty of the environments) and confirms that our CL-based method that uses ToL outperforms the E2E methodology. In particular, we improve the average success rate and the safety of the trained policy, resulting in 10\% fewer collisions in unseen testing scenarios. To further confirm these results, we employ a formal verification tool to quantify the number of correct behaviors of Reinforcement Learning policies over desired specifications.


## Setup the environment for this work
> To use this repo you must have a linux system because the environments have been compiled in Unity for that system. We suggest creating a virtual environment using Anaconda and installing all the dependencies reported above.

1. Download [Anaconda](https://www.anaconda.com/distribution/#download-section) for your System.

2.  Install Anaconda
	- On Linux/Mac 
		- Use *sh Anaconda{version}.sh* to install.
		- Add it to the PATH during installation if you’re ok with it:
			- First *export PATH=~/anaconda3/bin:$PATH*
			- Then *source ~/.bashrc*
		- *sudo apt-get install git* (may be required).

3.  Setup conda environment:
	- `
git clone https://github.com/LM095/Curriculum-Learning-for-Safe-Mapless-Navigation`
	- `cd Curriculum-Learning-for-Safe-Mapless-Navigation`
	- `conda env create -f CL-env.yml`
	
4.  If you want to delete this new conda environment:
	- `conda env remove -n cl-env`
	
	
In this work we use three different methodologies namely Transfer of Learning, fine-tuning, and finally, the classical E2E training. In this work, we consider the area occupied by obstacles and the smallest distance between two obstacles in the environment itself as a metric of difficulty. The biggest difficulties that we find in a navigation task are very close obstacles, narrow corridors, *U-shaped* walls. Hence, we insert all these complex components in a final environment, which we refer to as *finalEnv*, and then train the agent with three different methodologies.


## Training 
`conda activate cl-env`


>All the weights of the nets that you train with the various methods are automatically saved in the folder 
`Curriculum-Learning-for-Safe-Mapless-Navigation/PPO_preTrained/`

### Step 1: Train E2E
We start the training in the *finalEnv* environment used in this work with an E2E training (in an uninformed fashion) to see how many steps are required to get a high success rate.
```
cd Curriculum-Learning-for-Safe-Mapless-Navigation
python train_e2e.py
```
### Step 2: Train the agent using CL-based methodologies
In our results the steps required to reach the constant high success rate by the E2E methodology are 6M (see Fig. in results below), so we divide the steps for the informed methodology into respectively 1M steps for *baseEnv*, 2M for *intEnv*, and finally 3M steps for *finalEnv*


 -  **CL + Transfer of Learning**
 
	 At line 125 of train_CL.py file you can choose the methodology for the training. In order to 	
	 train the agent with CL + ToL, you should change `method = 'FineTuning'` into `method = 	
	 'TransferLearning'`. 
	```
	cd Curriculum-Learning-for-Safe-Mapless-Navigation
	python train_CL.py
	```
 -  **CL + Fine-tuning**

	 At line 125 of train_CL.py file you can choose the methodology for the training. In order to 	
	 train the agent with CL + ToL, you should change `method = 'TransferLearning'` into 		
	 `method = 'FineTuning'`. 
	```
	cd Curriculum-Learning-for-Safe-Mapless-Navigation
	python train_CL.py
	```
	
## Results training:
![](https://i.imgur.com/EJckV7m.png)
> ( a ) E2E plot. ( b ) Plot comparison of ToF and Fine-tuning. The value reported is the median of the best performance (success rate) of all randomly seeded runs of each method. The first peak, for the informed training, denotes the completion of training in the *baseEnv*, the second peak denotes the completion of training in the *intEnv*, and similarly, the third peak denotes the completion in the *finalEnv*. ( c ) Comparison between E2E and ToL: starting and goal positions are randomly chosen, so they may not coincide at the same timestep.



### Step 3: Test the agent on unseen environments
The test is performed on five new unseen environments besides the one where we trained the robot. These new scenarios all have similar characteristics to the *finalEnv*, and to establish the difficulty level, we define two metrics: the area occupied by obstacles and the minimum distance between two obstacles in the environment.

If you want to use our pre-trained models for the test phase, you will find them in the folder `Curriculum-Learning-for-Safe-Mapless-Navigation/PPO_preTrained/` under the name of `PPO_E2E.pth`, `PPO_FineTuning.pth`, `PPO_TransferLearning.pth`.
If you have trained your networks, make sure the .pth file names are the same as written above.

At line 28 of `test.py` file you can choose the environment for the testing phase. You can choose among these envs (depicted below): `finalEnv`, `envTest1`, `envTest2`, `envTest3`, `envTest4`, `envTest5`.

	cd Curriculum-Learning-for-Safe-Mapless-Navigation
	python test.py
	

## Results testing:
![](https://i.imgur.com/YWJU12s.png)
<img src="https://i.imgur.com/P0WsSG8.png" width="450" height="400">


## ACKNOWLEDGMENTS
The research has been partially supported by the projects ”Dipartimenti di Eccellenza 2018-2022”, funded by the Italian Ministry of Education, Universities and Research(MIUR).


