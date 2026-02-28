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

import torch
from scene import Scene, DeformModel, MlpModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_mytime(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform, mlp_model, dataset):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_scene = []
    render_obj_2 = []
    render_feature = []
    render_segmented = []
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        output = render(view, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        
        
        image = output["render"] #[3,H,W]
        language_feature = output["language_feature_image"]  #[3,H,W]
        gt = view.original_image[0:3, :, :]


        N = language_feature.shape[1] * language_feature.shape[2]
        language_feature_reshaped = language_feature.permute(1, 2, 0).reshape(N, -1)
        obj_id_distribution = mlp_model.step(language_feature_reshaped) # [N,3] -> [N,4] (3 is latent embeedding, 4 is number of classes including no relevant object)
        obj_id_prob, obj_id= obj_id_distribution.max(dim=1) #[N,4] -> [N], obj_id in [0,1,2,3], 3 means no relevant object
        obj_id = obj_id.reshape(language_feature.shape[1], language_feature.shape[2]) # [H,W]
        mask_obj_2 = (obj_id == 2) # mask for object id 2 [H,W]
        # Clone image and set non-object-2 pixels to white
        image_obj_2 = image.clone()
        mask_broadcast = ~mask_obj_2.unsqueeze(0).expand_as(image)  # [3,H,W]
        image_obj_2[mask_broadcast] = 1.0
        
        number_of_class = obj_id_distribution.shape[1] #4
        feature = obj_id.float()[None, :, :]  # [1,H,W]
        feature = feature / (number_of_class - 1)

        image_segmented = image.clone()
        mask_obj_0 = (obj_id == 0)
        mask_obj_1 = (obj_id == 1)
        alpha = 0.5
        # Blend object 0 with red
        red = torch.tensor([1.0, 0.0, 0.0], device=image_segmented.device).view(3, 1, 1)
        #mask_broadcast_0 = mask_obj_0.unsqueeze(0).expand_as(image_segmented)
        image_segmented = torch.where(mask_obj_0, alpha * image_segmented + (1 - alpha) * red, image_segmented)
        # Blend object 1 with green
        green = torch.tensor([0.0, 1.0, 0.0], device=image_segmented.device).view(3, 1, 1)
        #mask_broadcast_1 = mask_obj_1.unsqueeze(0).expand_as(image_segmented)
        image_segmented = torch.where(mask_obj_1, alpha * image_segmented + (1 - alpha) * green, image_segmented)
        #Blend object 2 with blue
        blue = torch.tensor([0.0, 0.0, 1.0], device=image_segmented.device).view(3, 1, 1)
        #mask_broadcast_2 = mask_obj_2.unsqueeze(0).expand_as(image_segmented)
        image_segmented = torch.where(mask_obj_2, alpha * image_segmented + (1 - alpha) * blue, image_segmented)

        # torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        render_scene.append(to8b(image.cpu().numpy()))
        render_obj_2.append(to8b(image_obj_2.cpu().numpy()))
        render_feature.append(to8b(feature.cpu().numpy()))
        render_segmented.append(to8b(image_segmented.cpu().numpy()))

    render_scene = np.stack(render_scene, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'render_scene.mp4'), render_scene, fps=30, quality=8)
    render_obj_2 = np.stack(render_obj_2, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'render_obj_2.mp4'), render_obj_2, fps=30, quality=8)
    render_feature = np.stack(render_feature, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'render_feature.mp4'), render_feature, fps=30, quality=8)
    render_segmented = np.stack(render_segmented, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'render_segmented.mp4'), render_segmented, fps=30, quality=8)   

#create myview. It should be same to mytime but only render a single fixed view (middle view in the list)
def render_myview(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform, mlp_model, dataset):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    view_to_render = views[len(views) // 2]  # Choose the middle view for rendering
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_scene = []
    render_obj_2 = []
    render_feature = []
    render_segmented = []
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        output = render(view_to_render, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        
        
        image = output["render"] #[3,H,W]
        language_feature = output["language_feature_image"]  #[3,H,W]
        gt = view.original_image[0:3, :, :]


        N = language_feature.shape[1] * language_feature.shape[2]
        language_feature_reshaped = language_feature.permute(1, 2, 0).reshape(N, -1)
        obj_id_distribution = mlp_model.step(language_feature_reshaped) # [N,3] -> [N,4] (3 is latent embeedding, 4 is number of classes including no relevant object)
        obj_id_prob, obj_id= obj_id_distribution.max(dim=1) #[N,4] -> [N], obj_id in [0,1,2,3], 3 means no relevant object
        obj_id = obj_id.reshape(language_feature.shape[1], language_feature.shape[2]) # [H,W]
        mask_obj_2 = (obj_id == 2) # mask for object id 2 [H,W]
        # Clone image and set non-object-2 pixels to white
        image_obj_2 = image.clone()
        mask_broadcast = ~mask_obj_2.unsqueeze(0).expand_as(image)  # [3,H,W]
        image_obj_2[mask_broadcast] = 1.0
        
        number_of_class = obj_id_distribution.shape[1] #4
        feature = obj_id.float()[None, :, :]  # [1,H,W]
        feature = feature / (number_of_class - 1)

        image_segmented = image.clone()
        mask_obj_0 = (obj_id == 0)
        mask_obj_1 = (obj_id == 1)
        alpha = 0.5
        # Blend object 0 with red
        red = torch.tensor([1.0, 0.0, 0.0], device=image_segmented.device).view(3, 1, 1)
        #mask_broadcast_0 = mask_obj_0.unsqueeze(0).expand_as(image_segmented)
        image_segmented = torch.where(mask_obj_0, alpha * image_segmented + (1 - alpha) * red, image_segmented)
        # Blend object 1 with green
        green = torch.tensor([0.0, 1.0, 0.0], device=image_segmented.device).view(3, 1, 1)
        #mask_broadcast_1 = mask_obj_1.unsqueeze(0).expand_as(image_segmented)
        image_segmented = torch.where(mask_obj_1, alpha * image_segmented + (1 - alpha) * green, image_segmented)
        #Blend object 2 with blue
        blue = torch.tensor([0.0, 0.0, 1.0], device=image_segmented.device).view(3, 1, 1)
        #mask_broadcast_2 = mask_obj_2.unsqueeze(0).expand_as(image_segmented)
        image_segmented = torch.where(mask_obj_2, alpha * image_segmented + (1 - alpha) * blue, image_segmented)

        # torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        render_scene.append(to8b(image.cpu().numpy()))
        render_obj_2.append(to8b(image_obj_2.cpu().numpy()))
        render_feature.append(to8b(feature.cpu().numpy()))
        render_segmented.append(to8b(image_segmented.cpu().numpy()))

    render_scene = np.stack(render_scene, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'render_scene.mp4'), render_scene, fps=30, quality=8)
    render_obj_2 = np.stack(render_obj_2, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'render_obj_2.mp4'), render_obj_2, fps=30, quality=8)
    render_feature = np.stack(render_feature, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'render_feature.mp4'), render_feature, fps=30, quality=8)
    render_segmented = np.stack(render_segmented, 0).transpose(0,2,3,1)
    imageio.mimwrite(os.path.join(render_path, 'render_segmented.mp4'), render_segmented, fps=30, quality=8)
def render_mycull(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform, mlp_model, dataset):
    
    gaussian_language_feature = gaussians.get_language_feature # [N,16]
    gaussian_obj_id_distribution = mlp_model.step(gaussian_language_feature) # [N,16] -> [N,4]
    #gaussian_obj_id_prob, gaussian_obj_id= gaussian_obj_id_distribution.max(dim=1) #[N,4] -> [N], obj_id in [0,1,2,3], 3 means no relevant object
    #mask_gaussian_obj_2 = (gaussian_obj_id == 2) # mask for object id 2 [N]
    #calculate probability with softmax
    gaussian_obj_id_prob = torch.softmax(gaussian_obj_id_distribution, dim=1) # [N,4]
    mask_gaussian_obj_2 = (gaussian_obj_id_prob[:,2] > 0.3) # mask for object id 2 [N]
    #print number of selected gaussians
    print("Number of gaussians selected for object 2: {}/{}".format(mask_gaussian_obj_2.sum().item(), mask_gaussian_obj_2.shape[0]))



    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    renderings = []
    gts = []
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        output = render(view, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh, cull_mask=mask_gaussian_obj_2)
        
        
        image = output["render"] #[3,H,W]
        language_feature = output["language_feature_image"]  #[3,H,W]

        gt = view.original_image[0:3, :, :]
        if args.train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        renderings.append(to8b(image.cpu().numpy()))
        gts.append(to8b(gt.cpu().numpy()))
    renderings = np.stack(renderings, 0).transpose(0,2,3,1) #[Frame,H,W,3]
    imageio.mimwrite(os.path.join(render_path, 'mytime_video_rendering.mp4'), renderings, fps=30, quality=8)
    gts = np.stack(gts, 0).transpose(0,2,3,1) #[Frame,H,W,3]
    imageio.mimwrite(os.path.join(render_path, 'mytime_video_gt.mp4'), gts, fps=30, quality=8)

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform, mlp_model, dataset):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        output = render(view, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        
        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]  #[3,H,W]

        if not args.include_feature:
            gt = view.original_image[0:3, :, :]
            if args.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]
        else:
            gt, mask = view.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=dataset.feature_level)

        if args.include_feature:
            N = rendering.shape[1] * rendering.shape[2]
            language_feature_reshaped = rendering.permute(1, 2, 0).reshape(N, -1)
            obj_id_distribution = mlp_model.step(language_feature_reshaped) # [N,3] -> [N,4] (3 is latent embeedding, 4 is number of classes including no relevant object)
            obj_id_prob, obj_id= obj_id_distribution.max(dim=1) #[N,4] -> [N], obj_id in [0,1,2,3], 3 means no relevant object

            num_of_classes = obj_id_distribution.shape[1] #4 = number of positives + 1 (no relevant object)
            rendering = obj_id.reshape(rendering.shape[1], rendering.shape[2])[None, :, :].float()  # [1,H,W]
            # Normalize rendering to [0, 1], the max value is num_of_classes -1
            rendering = rendering / (num_of_classes - 1)

            #process gt with the mask to set background points to "no relevant object" class (i.e., 3)
            gt = gt * mask + (num_of_classes - 1) * (~mask)  # [1,H,W]
            gt = gt.float() / (num_of_classes - 1)  # normalize gt to [0,1]


            # language_feature_reshaped = rendering.permute(1, 2, 0).reshape(N, 3) # [N,3]
            # obj_id_distribution = mlp_model.step(language_feature_reshaped) # [N,3] -> [N,3] (the 1st 3 is latent embeedding, the 2nd 3 is number of objects )
            # obj_id_prob, obj_id= obj_id_distribution.max(dim=1) #[N,3] -> [N]
            # obj_mask = mask.permute(1, 2, 0).reshape(N)  # [N], mask to remove background point (i.e., no mask point)
            # #valid_mask = obj_id_prob > 0.5 # threshold to remove irrrelevant postive id
            # obj_id *= obj_mask 

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def interpolate_time(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform, mlp_model, dataset):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    #view = views[idx]
    view = views[len(views) // 2]  # Choose the middle view for rendering
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        if not args.include_feature:
            rendering = results["render"]
        else:
            rendering = results["language_feature_image"]
            rendering = torch.clamp((rendering+1)/2, 0.0, 1.0)
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform, mlp_model, dataset):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    #view = views[idx]  # Choose a specific time for rendering
    view = views[len(views) // 2]  # Choose the middle view for rendering
    #view = views[10]  # Choose the first view for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid.cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(acc, os.path.join(acc_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform, mlp_model, dataset):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, 60.0, 30.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]], #default is -30.0, 4.0
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform):    
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

        results = render(view, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)

def interpolate_view_original(model_path, source_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args, load2gpu_on_the_fly, is_6dof, deform):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

        results = render(view, gaussians, d_xyz, d_rotation, d_scaling, is_6dof, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        #scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt40000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, dataset, mode='test')
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)
        mlp_model = None
        if args.include_feature:
            mlp_model = MlpModel(input_size=dataset.langauge_feautre_dim, output_size=4)
            mlp_model.load_weights(dataset.model_path)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.mode == "render":
            render_func = render_set
        elif args.mode == "time":
            render_func = interpolate_time
        elif args.mode == "view":
            render_func = interpolate_view
        elif args.mode == "pose":
            render_func = interpolate_poses
        elif args.mode == "original":
            render_func = interpolate_view_original
        elif args.mode == "mytime":
            render_func = render_mytime
        elif args.mode == "mycull":
            render_func = render_mycull
        elif args.mode == "myview":
            render_func = render_myview
        else:
            render_func = interpolate_all
        if not skip_train and scene.getTrainCameras():
             #render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args, dataset.load2gpu_on_the_fly, dataset.is_6dof, deform)
            render_func(dataset.model_path, dataset.source_path, args.mode, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args, dataset.load2gpu_on_the_fly, dataset.is_6dof, deform, mlp_model, dataset)
        if not skip_test and scene.getTestCameras():
             #render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args, dataset.load2gpu_on_the_fly, dataset.is_6dof, deform)
             render_func(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args, dataset.load2gpu_on_the_fly, dataset.is_6dof, deform, mlp_model, dataset)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--include_max_contrib", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original', 'mytime', 'myview', 'mycull'])
    #parser.add_argument("--language_features_name", type=str, default="language_features.npy")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args)