CUDA_VISIBLE_DEVICES=1 python main.py --ckpt ckpt/best_model_custom.pth.tar --criterion custom
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_mse.pth.tar --criterion mse
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_ssim.pth.tar --criterion ssim
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_bce.pth.tar --criterion bce
CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_l1.pth.tar --criterion l1
