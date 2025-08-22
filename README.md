# tinydiff: A tiny diffusion training example

Reference:
  - smalldiffusion: https://github.com/yuanchenyang/smalldiffusion
  - https://sander.ai/2022/05/26/guidance.html

## Classifier-free guidance
Train a conditional diffusion model `p(x|y)` with conditional dropout: some percentage of time(10-20% tends to work well), the conditioning information `y` is removed. In practice, it is often replaced with a special input value representing the absence of conditioning information. The resulting model is now able to function both as conditional model `p(x|y)` and as an unconditional model `p(x)` depending on whether the conditioning signal is provided. 
