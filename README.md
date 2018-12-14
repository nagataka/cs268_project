# README

This is CS268 (Intro to Optimization) project of J.B. Lanier and Takashi Nagata.


#### Python environment

* Make sure that you have not installed [OpenAI baselines](https://github.com/openai/baselines) in your environment yet (we need a specific version of that) or,
* Create a new virtual environment with python3.5 or later
    * If you are using [conda](https://conda.io/docs/index.html), do the following
    * conda create -n cs268 python=3.5

#### Clone and install the project dependencies

```sh
git clone https://github.com/nagataka/cs268_project.git
cd cs268_project
pip install numpy
pip install -e baselines
cd ppo_implementation
```

#### Run demo!

Just run the script we made to use pre-trained network on Breakout-v4.

```sh
sh demo_atari.sh
```

If you want to train our PPO implementation, run

```
train_on_atari.sh
```

#### Mujoco

While Atari runs out of the box, to run HalfCheetah-v2 demo (demo_mujoco.sh) you need Mujoco which is a proprietary software and you need to get a license in here (Personal 1 year license for free or 30 days trial):
[https://www.roboti.us/license.html](https://www.roboti.us/license.html)

After get a license, please follow the instructions here: [mujoco-py](https://github.com/openai/mujoco-py)

Finally, modify baselines/setup.py line 28  

Before
```python
gym[atari,classic_control]
```
After
```python
gym[atari,classic_control,mujoco]
```