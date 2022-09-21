This repository contain depth estimation code and validated on the train datasets of the challenge [SimCol-to-3D 2022 - 3D Reconstruction During Colonoscopy](https://www.synapse.org/#!Synapse:syn28548633/wiki/617126)
#MSE: MSE:0.000265, SSIM:0.968876
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_mse.pth.tar --criterion mse

# SSIM: MSE:0.001259, SSIM:0.966289
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_ssim.pth.tar --criterion ssim

# BCE:MSE:0.000208, SSIM:0.972940
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_bce.pth.tar --criterion bce 

# L1: MSE:0.000115, Best SSIM:0.984670
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_l1.pth.tar --criterion l1

# Custom: MSE:0.000257, SSIM:0.969468
CUDA_VISIBLE_DEVICES=1 python main.py --ckpt ckpt/best_model_custom.pth.tar --criterion custom


# MSE: MSE:0.001521, SSIM:0.931749
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_mse_aug.pth.tar --aug --criterion mse

# SSIM: 
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_ssim_aug.pth.tar --aug --criterion ssim

# L1: MSE:0.001124, SSIM:0.951270
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_l1_aug.pth.tar --aug --criterion l1

# BCE: MSE:0.001174, SSIM:0.944775
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_bce_aug.pth.tar --aug --criterion bce

# 
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_custom_aug.pth.tar--aug --criterion custom
