# Curriculum Learning for Safe Mapless Navigation

 **[Luca Marzari](https://github.com/LM095), Davide Corsi, Enrico Marchesini and Alessandro Farinelli.** 

## Abstract
This work investigates the effects of Curriculum Learning (CL)-based approaches on the agent's performance. In particular, we focus on the safety aspect of robotic mapless navigation, comparing over a standard end-to-end (E2E) training strategy. To this end, we present a CL approach that leverages Transfer of Learning (ToL) and fine-tuning in a Unity-based simulation with the Robotnik Kairos as a robotic agent. For a fair comparison, our evaluation considers an equal computational demand for every learning approach (i.e., the same number of interactions and difficulty of the environments) and confirms that our CL-based method that uses ToL outperforms the E2E methodology. In particular, we improve the average success rate and the safety of the trained policy, resulting in 10\% fewer collisions in unseen testing scenarios. To further confirm these results, we employ a formal verification tool to quantify the number of correct behaviors of Reinforcement Learning policies over desired specifications.


## Implementation details
### Prerequisites
- Python3.7+
- PyTorch
- OpenAI gym
- gym-env==0.16.1
- mlagents==0.16.1

>We suggest creating a virtual environment using Anaconda and installing all the dependencies reported above.


In this work we use three different methodologies namely Transfer of Learning, fine-tuning, and finally, the classical E2E training. In this work, we consider the area occupied by obstacles and the smallest distance between two obstacles in the environment itself as a metric of difficulty. The biggest difficulties that we find in a navigation task are very close obstacles, narrow corridors, *U-shaped* walls. Hence, we insert all these complex components in a final environment, which we refer to as *finalEnv*, and then train the agent with three different methodologies.

### Step 0: Clone the repository
```
git clone https://github.com/LM095/Curriculum-Learning-for-Safe-Mapless-Navigation
```

## Training 

All the weights of the nets that you train with the various methods are automatically saved in the folder 
`Curriculum-Learning-for-Safe-Mapless-Navigation/PPO_preTrained/`

### Step 1: Train E2E
We start the training in the *finalEnv* environment used in this work with an E2E training (in an uninformed fashion) to see how many steps are required to get a high success rate.
```
cd Curriculum-Learning-for-Safe-Mapless-Navigation
python train_e2e.py
```
### Step 2: Train the agent using CL-based methodologies
In our results the steps required to reach the constant high success rate by the E2E methodology are 6M (see Fig. in results below), so we divide the steps for the informed methodology into respectively 1M steps for *baseEnv*, 2M for *intEnv*, and finally 3M steps for *finalEnv*

#### 

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



## Testing 
If you want to use our pre-trained models for the test phase, you will find them in the folder `Curriculum-Learning-for-Safe-Mapless-Navigation/PPO_preTrained/` under the name of 
