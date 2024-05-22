import logging
import os
import torch
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from srcs.utils.util import instantiate
from srcs.utils.utils_image_kair import tensor2uint, imsave

def testing(gpus, config):
    test_worker(gpus, config)


def test_worker(gpus, config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # logger & dir setting
    logger = logging.getLogger('test')
    os.makedirs(config.outputs_dir,exist_ok=True)

    # prepare model & checkpoint for testing
    # load checkpoint
    logger.info(f"💡 Loading checkpoint: {config.checkpoint} ...")
    checkpoint = torch.load(config.checkpoint)
    logger.info("💡 Checkpoint loaded!")

    # select config file
    if 'config' in checkpoint:
        loaded_config = OmegaConf.create(checkpoint['config'])
    else:
        loaded_config = config

    # instantiate model
    model = instantiate(loaded_config.arch)
    logger.info(model)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)

    # load weight
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # instantiate loss and metrics
    criterion=None # don't calc loss in test
    metrics = [instantiate(met) for met in config.metrics]

    # setup data_loader instances
    data_loader = instantiate(config.test_data_loader)
    
    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(data_loader, model,
               device, criterion, metrics, config, logger)
    logger.info(log)


def test(data_loader, model,  device, criterion, metrics, config, logger=None):
    '''
    test step
    '''
    # init
    model = model.to(device)
    interp_scale = getattr(model, 'frame_n', 8)//getattr(model, 'ce_code_n', 8)
    if config.get('save_img', False):
        os.makedirs(config.outputs_dir+'/output')
        os.makedirs(config.outputs_dir+'/target')
        os.makedirs(config.outputs_dir+'/input')

    # run
    ce_weight = model.BlurNet.ce_weight.detach().squeeze()
    ce_code = ((torch.sign(ce_weight)+1)/2).int()
    model.eval()
    total_metrics = torch.zeros(len(metrics), device=device)
    time_start = time.time()
    with torch.no_grad():
        for i, vid in enumerate(tqdm(data_loader, desc='⏳ Testing')):
            # move vid to gpu, convert to 0-1 float
            vid = vid.to(device).float()/255 
            N, F, C, Hx, Wx = vid.shape

            # direct
            output, data, data_noisy = model(vid)

            # clamp to 0-1
            output = torch.clamp(output, 0, 1)

            # save some sample images
            if config.get('save_img', False):
                scale_fc = len(ce_code)/sum(ce_code)
                for k, (in_img, out_img, gt_img) in enumerate(zip(data, output, vid)):
                    in_img = tensor2uint(in_img*scale_fc)
                    imsave(
                        in_img, f'{config.outputs_dir}input/ce-blur#{i*N+k+1:04d}.jpg')
                    for j in range(len(ce_code)):
                        out_img_j = tensor2uint(out_img[j])
                        gt_img_j = tensor2uint(gt_img[j])
                        imsave(
                            out_img_j, f'{config.outputs_dir}output/out-frame#{i*N+k+1:04d}-{j+1:04d}.jpg')
                        imsave(
                            gt_img_j, f'{config.outputs_dir}target/gt-frame#{i*N+k+1:04d}-{j+1:04d}.jpg')

            # metrics on test set
            output_all = torch.flatten(output, end_dim=1)
            target_all = torch.flatten(vid[:,::interp_scale], end_dim=1)
            batch_size = data.shape[0]
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output_all, target_all) * batch_size
    time_end = time.time()
    time_cost = time_end-time_start
    n_samples = len(data_loader.sampler)
    log = {'time/sample': time_cost/n_samples,
           'ce_code': ce_code}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    return log
