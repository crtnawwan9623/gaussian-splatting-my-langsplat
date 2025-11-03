#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel, MlpModel
from utils.general_utils import safe_state, get_expon_lr_func, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
def sum_params_from_optimizer(optimizer):
    if optimizer is None or not hasattr(optimizer, "param_groups"):
        return float('nan')
    with torch.no_grad():
        total = 0.0
        for group in optimizer.param_groups:
            for p in group.get('params', []):
                if p is not None:
                    total += p.data.sum().item()
        return total
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    mlp_model = None
    if opt.include_feature:
        mlp_model = MlpModel()
        mlp_model.train_setting(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        # if len(model_params) == 12 and opt.include_feature:
        #     first_iter = 0
        if opt.include_feature:
            if len(model_params) == 12:
                first_iter = 0
            else:
                mlp_model.load_weights(dataset.model_path)
        gaussians.restore(model_params, opt)
        deform.load_weights(dataset.model_path)
        # if opt.include_feature:
        #     mlp_model.load_weights(dataset.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    Ll1depth = 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, d_xyz, d_rotation, d_scaling, dataset.is_6dof, pipe, background, opt, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if not opt.include_feature:
            deform.update_learning_rate(iteration)
        if opt.include_feature:
            mlp_model.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
            total_frame = len(viewpoint_stack)
            time_interval = 1 / total_frame
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
		
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
			
        fid = viewpoint_cam.fid
        
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
			
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, d_xyz, d_rotation, d_scaling, dataset.is_6dof, pipe, bg, opt, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, language_feature, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["language_feature_image"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        if opt.include_feature:
            gt_language_feature, language_feature_mask = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=dataset.feature_level)
            #Ll1 = l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)
            #reshape langauge_feautre from [3,H,W] to [N,3]
            N = language_feature.shape[1] * language_feature.shape[2]
            language_feature_reshaped = language_feature.permute(1, 2, 0).reshape(N, 3)
            obj_id_distribution = mlp_model.step(language_feature_reshaped) # [N,3] -> [N,4] (3 is latent embeedding, 4 is number of classes including no relevant object)
            obj_id = gt_language_feature # [1, H, W], possible values are 0,1,2,3 (len(positives)=3 means no relevant object)
            obj_mask = language_feature_mask # [1, H, W]
            #calculate cross entropy loss of obj_id_distribution and obj_id, considering obj_mask
            obj_id = obj_id.permute(1, 2, 0).reshape(N).long()  # [N]

            # mask to ignore background. 
            # obj_mask=language_feature_mask is grouped mask (0/1) for all masks in the image. 
            # Background points don't belong to any mask, and have seg map index -1. 
            # Hence obj_mask=0 for background points (calculated in get_language_feature)
            obj_mask = obj_mask.permute(1, 2, 0).reshape(N)  # [N]

            # #second mask to ignore non-relevant objects in foreground
            # # object id -1 (non-relevant objects)
            # valid_mask = obj_id != -1  # [N]
            # #combine two masks
            # obj_mask = obj_mask * valid_mask  # [N]

            #For background points (mask=0), use len(positives)=3 as the ground truth label (same as no relevant object)
            #this way, we can use all points for training, and not ignore background points
            #obj_id[~obj_mask] = obj_id_distribution.shape[1] - 1  # [N], set background points to "no relevant object" class (i.e., 3)
            
            #create mask to ignore irrelevant object id (3)
            valid_obj_mask = (obj_id >= 0) & (obj_id < obj_id_distribution.shape[1]-1)  # [N]
            obj_mask = obj_mask & valid_obj_mask  # [N]
            #set background and irrelevant object id to no relevant object class(3)
            obj_id[~obj_mask] = obj_id_distribution.shape[1] - 1

            assert obj_id.min() >= 0 and obj_id.max() < obj_id_distribution.shape[1]
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            ce_loss = criterion(obj_id_distribution, obj_id)  # [N]
            # #calculate mean cross entropy loss for relevant object points
            # ce_loss_mean = (ce_loss * obj_mask).sum() / (obj_mask.sum() + 1e-8)
            # #calculate mean cross entropy loss for background and irrelevant object points
            # ce_loss_mean_back_and_irrelevant = (ce_loss * (~obj_mask)).sum() / ((~obj_mask).sum() + 1e-8)
            ce_loss_mean = ce_loss.mean()
            Ll1 = ce_loss_mean
            # lambda_ce_back_and_irrelevant = 0.3
            # Ll1 = (1- lambda_ce_back_and_irrelevant) * ce_loss_mean + lambda_ce_back_and_irrelevant * ce_loss_mean_back_and_irrelevant
            loss = Ll1

            # lambda_ce_back_and_irrelevant = 0.5
            # pos = obj_mask
            # neg = ~pos
            # pos_count = pos.sum().clamp(min=1)
            # neg_count = neg.sum().clamp(min=1)

            # w_pos = (1.0 - lambda_ce_back_and_irrelevant) / pos_count
            # w_neg = lambda_ce_back_and_irrelevant / neg_count
            # weights = torch.where(pos, w_pos, w_neg) # [N]

            # Ll1 = (weights * ce_loss).sum()  # total weight ~ 1 each step
            # loss = Ll1
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # if opt.include_feature:
            #     mlp_model.mlp.eval()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
							testing_iterations, scene, render, (pipe, background, opt, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset, deform,
							dataset.load2gpu_on_the_fly, mlp_model)
            # if opt.include_feature:
            #     mlp_model.mlp.train()
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                #deform.save_weights(args.model_path, iteration)

            # Densification
            if not opt.include_feature:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none = True)
            if use_sparse_adam:
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)
            else:
                param_sum = sum_params_from_optimizer(gaussians.optimizer)
                print(f"Gaussians parameter sum before step: {param_sum}") if iteration % 100 == 0 else None
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                param_sum = sum_params_from_optimizer(gaussians.optimizer)
                print(f"Gaussians parameter sum after step: {param_sum}") if iteration % 100 == 0 else None

                if not opt.include_feature:
                    deform.optimizer.step()
                    deform.optimizer.zero_grad()
                freeze_mlp_until_iter = 1000
                if opt.include_feature and iteration < freeze_mlp_until_iter:
                    # calculate pre_sum of mlp_model parameters for debugging
                    param_sum = sum_params_from_optimizer(mlp_model.optimizer)
                    print(f"MLP model parameter sum before step: {param_sum}") if iteration % 100 == 0 else None
                    mlp_model.optimizer.step()
                    mlp_model.optimizer.zero_grad()
                    param_sum = sum_params_from_optimizer(mlp_model.optimizer)
                    print(f"MLP model parameter sum after step: {param_sum}") if iteration % 100 == 0 else None

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if not opt.include_feature:
                    deform.save_weights(args.model_path, iteration)
                if opt.include_feature:
                    mlp_model.save_weights(args.model_path, iteration)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, 
					renderArgs, dataset, deform, load2gpu_on_the_fly, mlp_model):
    opt = renderArgs[2]
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    if not opt.include_feature:
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, d_xyz, d_rotation, d_scaling, dataset.is_6dof, *renderArgs)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if dataset.train_test_exp:
                            image = image[..., image.shape[-1] // 2:]
                            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    else:
                        gt_image, mask = viewpoint.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=dataset.feature_level) # [1,H,W]

                        rendering = renderFunc(viewpoint, scene.gaussians, d_xyz, d_rotation, d_scaling, dataset.is_6dof, *renderArgs)["language_feature_image"] # [3,H,W]
                        language_feature_reshaped = rendering.permute(1, 2, 0).reshape(-1, 3) # [N,3]
                        obj_id_distribution = mlp_model.step(language_feature_reshaped) # [N,3] -> [N,4]

                        gt_image = gt_image.to("cuda")
                        gt_image = gt_image * mask + (~mask) * (obj_id_distribution.shape[1] - 1)  # set non-relevant object points to "no relevant object" class (i.e., 3)
                        #normalize to [0,1]
                        gt_image = gt_image.float() / (obj_id_distribution.shape[1] - 1)

                        obj_id_prob, obj_id= obj_id_distribution.max(dim=1) # [N], possible values of obj_id are 0,1,2,3
                        obj_mask = mask.permute(1, 2, 0).reshape(-1)  # [N]
                        #assert that obj_mask has same size as obj_id
                        if obj_id.shape[0] != obj_mask.shape[0]:
                            raise ValueError(f"obj_id size {obj_id.shape[0]} does not match obj_mask size {obj_mask.shape[0]}")
                        #obj_id[~obj_mask] = obj_id_distribution.shape[1] - 1  # set background points to "no relevant object" class (i.e., 3)
                        image = obj_id.reshape(rendering.shape[1], rendering.shape[2]).unsqueeze(0)  # [1,H,W]
                        #normalize to [0,1]
                        image = image.float() / (obj_id_distribution.shape[1] - 1)

                        # mask = mask.permute(1, 2, 0).reshape(-1).to("cuda")  # [N]
                        # image = image.permute(1, 2, 0).reshape(-1)  # [N]
                        # gt_image = gt_image.permute(1, 2, 0).reshape(-1)  # [N]
                        # image = image[mask]
                        # gt_image = gt_image[mask]
                        # image = image.reshape(rendering.shape[1], rendering.shape[2]).unsqueeze(0) # [1,H,W]
                        # gt_image = gt_image.reshape(rendering.shape[1], rendering.shape[2]).unsqueeze(0) # [1,H,W] 

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        #if iteration == testing_iterations[1]:
                        if iteration == 2000:
                            #save images in text format
                            npy_path = os.path.join(scene.model_path, f"{config['name']}_obj_id_dist_N4_{viewpoint.image_name}_iter_{iteration}.txt")
                            npy_data = obj_id_distribution.cpu().numpy()
                            with open(npy_path, 'wb') as f:
                                import numpy as np
                                np.savetxt(f, npy_data, fmt='%.2f')

                            npy_path = os.path.join(scene.model_path, f"{config['name']}_feature_N3_{viewpoint.image_name}_iter_{iteration}.txt")
                            npy_data = language_feature_reshaped.cpu().numpy()
                            with open(npy_path, 'wb') as f:
                                import numpy as np
                                np.savetxt(f, npy_data, fmt='%.2f')
                    l1_test += l1_loss(image.float(), gt_image.float()).mean().double()
                    psnr_test += psnr(image.float(), gt_image.float()).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, 
						default=([1, 500, 7_000, 30_000] + list(range(1000, 40001, 1000))))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000, 40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    #args.model_path = args.model_path + f"_{str(args.feature_level)}"
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
