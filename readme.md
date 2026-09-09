# DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations [CVPR 2025]

新版情感 Avatar 默认使用 v3.2：状态相关的跨维度演化、持续双人反馈、90% 片段遮挡与 10% 整模态遮挡；保留同坐标快慢记忆和自主未来预测。实现与兼容约定见 [v3.2 修改说明](docs/v32_adaptive_dynamics_2026-09-09.md)，训练入口见 [docs/v3_training.md](docs/v3_training.md)。生成器每次输出 25 帧，原始上下文仍限于此前 3 秒。新旧训练 revision 不能混用优化器；旧 checkpoint 保留原构造推理。旧 v2 实验按 [docs/v2_training.md](docs/v2_training.md) 恢复。依赖见 `requirements-v2.txt`，CPU 回归测试见 `requirements-test.txt`。下方 Environment 保留原论文实现的环境说明。

Official PyTorch implementation for the paper:

> **DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations**, ***CVPR 2025***.
>
> Ziqiao Peng, Yanbo Fan, Haoyu Wu, Xuan Wang, Hongyan Liu, Jun He, Zhaoxin Fan
>
<p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2505.18096">Paper</a>
    | 
    <a href="https://ziqiaopeng.github.io/dualtalk/">Project Page</a>
    |
    <a href="https://github.com/ZiqiaoPeng/DualTalk">Code</a> 
  </b>
</p> 

<p align="center">
<img src="./media/DualTalk.png" width="95%" />
</p>

> Comparison of single-role models (Speaker-Only and Listener-Only) with DualTalk. Unlike single-role models, which lack key interaction elements, DualTalk supports speaking and listening role transition, multi-round conversations, and natural interaction.

## **Environment**

- Linux
- Python 3.6+
- Pytorch 1.12.1
- CUDA 11.3
- ffmpeg
- **[MPI-IS/mesh](https://github.com/MPI-IS/mesh)**	

Clone the repo:
  ```bash
  git clone https://github.com/ZiqiaoPeng/DualTalk.git
  cd DualTalk
  ```  
Create conda environment:
```bash
conda create -n dualtalk python=3.8.8
conda activate dualtalk
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
python install_pytorch3d.py
```
Before installation, you need to create an account on the [FLAME website](https://flame.is.tue.mpg.de/) and prepare your
login and password beforehand. You will be asked to provide them in the installation script.
Then, run the install.sh script to download the FLAME data and install the environment.
```bash
bash install.sh
```

## **Demo**

Download the pretrained model.

```bash
# If you are in China, you can set up a mirror.
# export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface-hub
huggingface-cli download ZiqiaoPeng/DualTalk --local-dir model
```


Given the audio and blendshape data, run:

```bash
python demo.py --audio1_path ./demo/xkHwlcDSOjc_sub_video_109_000_speaker2.wav --audio2_path ./demo/xkHwlcDSOjc_sub_video_109_000_speaker1.wav --bs2_path ./demo/xkHwlcDSOjc_sub_video_109_000_speaker1.npz
```

The results will be saved to `result_DualTalk` folder. 

## **Dataset**
Download the dataset from [DualTalk_Dataset](https://huggingface.co/datasets/ZiqiaoPeng/DualTalk_Dataset), unzip it, and place it in the data folder.
The data folder format is as follows:
- data
	- train
		- xxx.npz
		- xxx.wav
		- ...
	- test
		- xxx.npz
		- xxx.wav
		- ...
	- ood
		- xxx.npz
		- xxx.wav
		- ...



## **Training and Testing**

### Archived v1/v2 emotion-state rollout semantics

The emotion-state extension reports two distinct trajectory settings:

- `conditional` uses the observed future event/action sequence.
- `open_loop` uses the event/action available at the rollout origin, then calls
  `decay_only` with `DYNAMICS.OPEN_LOOP_DT`; it does not consume later events,
  actions, active-role values, or time intervals.

Use `configs/phase_a_dynamics.yaml`,
`configs/phase_a_dynamics_open_loop.yaml`, or
`configs/phase_a_dynamics_joint.yaml` to select conditional, open-loop, or
joint training.
The conditioned streaming configuration uses 25 frames (1 second at 25 FPS),
and chunk `k` is generated from the state produced after chunk `k-1`.
For the original 8-second, same-chunk offline ablation, override
`DUALTALK.CHUNK_FRAMES 200 DUALTALK.CAUSAL_STATE_CONTEXT False`.

### Training

- To train the model, run:

	```
	python main.py
	```

	You can find the trained models in `save_DualTalk` folder.

### Testing

- To test the model, run:

    ```
	python test.py
    ``` 

	The results will be saved to `result_DualTalk` folder.


### Visualization

- To visualize the results, run:

	```
	cd render
	python render_dualtalk_output.py
	```
	You can find the outputs in the `result_DualTalk` folder.

- To stitch the two speakers' video, run:

	```
	cd render
	python two_person_video_stitching.py
	```

### Evaluation

- To evaluate the model, run:

	```
	cd metric
	python metric.py
	```
	You can find the metrics results in the `metric` folder.

## **Citation**

If you find this code useful, please consider citing:

```bibtex
@inproceedings{peng2025dualtalk,
  title={Dualtalk: Dual-speaker interaction for 3d talking head conversations},
  author={Peng, Ziqiao and Fan, Yanbo and Wu, Haoyu and Wang, Xuan and Liu, Hongyan and He, Jun and Fan, Zhaoxin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21055--21064},
  year={2025}
}
```
