 CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model.pth.tar --criterion mse
 CUDA_VISIBLE_DEVICES=1 python main.py --ckpt ckpt/best_model.pth.tar --criterion custom
 CUDA_VISIBLE_DEVICES=1 python main.py --ckpt ckpt/best_model_custom.pth.tar --criterion custom

 CUDA_VISIBLE_DEVICES=0 python main.py --ckpt ckpt/best_model_mse.pth.tar --criterion mse
