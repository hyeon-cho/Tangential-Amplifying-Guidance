# [ICML2026] TAG: Tangential Amplifying Guidance for Hallucination-Resistant Diffusion Sampling
<a href="https://arxiv.org/abs/2510.04533"><img src="https://img.shields.io/badge/arXiv-2510.04533-%23B31B1B"></a>
<a href="https://hyeon-cho.github.io/TAG/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>  
<br>

This is the official implmentation of the paper *"TAG: Tangential Amplifying Guidance for Hallucination-Resistant Diffusion Sampling"*

by [Hyunmin Cho](https://github.com/hyeon-cho/)$^{\*}$, [Donghoon Ahn](https://sunovivid.github.io/)$^{\*}$, [Susung Hong](https://susunghong.github.io/)$^{\*}$, [Jee Eun Kim](https://github.com/jeeeun-k), [Seungryong Kim](https://cvlab.kaist.ac.kr/members/faculty)$^{†}$ and [Kyong Hwan Jin](https://ipa.korea.ac.kr/)$^{†}$

## Introduction 
![](assets/teaser.png)<br>
**Tangential Amplifying Guidance (TAG)** improves diffusion model sample quality by directly amplifying tangential components of estimated scores without modifying the model architecture.

## Runnable Code

You can test TAG with folloing commands:  
```
python run.py --config-name sd1
```
(or replace **`sd1`** with other config files)


## Citation
If you find this research useful, please consider citing:
```
@misc{cho2025tagtangentialamplifyingguidancehallucinationresistant,
      title={TAG:Tangential Amplifying Guidance for Hallucination-Resistant Diffusion Sampling}, 
      author={Hyunmin Cho and Donghoon Ahn and Susung Hong and Jee Eun Kim and Seungryong Kim and Kyong Hwan Jin},
      year={2025},
      eprint={2510.04533},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.04533}, 
}
```
