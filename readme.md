# MDM-2-DiffGAN implementation. Our [***paper***](https://github.com/CAP6412-Group-4/mdm-2-ddgan-report) #


## Set up dataset ##
**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

## Pretrained Weights ##
Download [humanml-encoder-512](https://drive.google.com/file/d/1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821/view?usp=sharing), ad save to 
```save``` folder.

## Training ##
To train our model, use the following script.

```shell
python -m train_ddgan --dataset humanml --num_channels 263 --batch_size 32
```

### Generation ##

### To generate a single prompt:
```shell
python -m sample.generate --dataset humanml --output_dir ./save/epoch325/toolbox --exp experiment --epoch_id 325 --text_prompt "the person walked forward and is picking up his toolbox."
```

### To generate from test set prompts

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_samples 10 --num_repetitions 3
```
## Evaluation ##

```shell
python -m eval.eval_humanml --dataset humanml --model_path saved_info/dd-gan/humanml/experiment/netG_325 --eval_mode mm_short --output_dir ./save --exp experiment --epoch_id 325 --node_rank 0 --text_prompt "A person jumping"
```

## License ##

Our code is distributed under both an [***MIT***](https://github.com/CAP6412-Group-4/denoising-diffusion-gan/blob/main/LICENSE) and [***NVIDIA***](https://github.com/CAP6412-Group-4/denoising-diffusion-gan/blob/main/LICENSE) License.