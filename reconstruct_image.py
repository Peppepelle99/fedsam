"""Run reconstruction in a terminal prompt.

Optional arguments can be found in inversefed/options.py
"""

import torch
import torchvision

import numpy as np
from PIL import Image

import inversefed

from collections import defaultdict
import datetime
import time
import os
from models.cifar100.resnet20 import ClientModel

torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

# Parse input arguments
args = inversefed.options().parse_args()
# Parse training strategy
defs = inversefed.training_strategy("conservative")
defs.epochs = args.epochs
# 100% reproducibility?
if args.deterministic:
    inversefed.utils.set_deterministic()


if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Prepare for training

    # Get data:
    # Serve mettere il cifar o lasciamo questo?
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path)

    dm = torch.as_tensor(getattr(inversefed.consts, f"{args.dataset.lower()}_mean"), **setup)[:, None, None]
    ds = torch.as_tensor(getattr(inversefed.consts, f"{args.dataset.lower()}_std"), **setup)[:, None, None]

    model = ClientModel(0.1, 10, 'GPU') # lr, n_classes da cambiare, device
    model_seed = None
    model.to(**setup)
    model.eval()

    #Num images = 1
    if args.target_id == -1:  # demo image
        # Specify PIL filter for lower pillow versions
        ground_truth = torch.as_tensor(
            np.array(Image.open("auto.jpg").resize((32, 32), Image.BICUBIC)) / 255, **setup
        )
        ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
        if not args.label_flip:
            labels = torch.as_tensor((1,), device=setup["device"])
        else:
            labels = torch.as_tensor((5,), device=setup["device"])
        target_id = -1
    else:
        #Se il target id è none prende un immagine a caso altrimenti quella dell'id scelto
        if args.target_id is None:
            target_id = np.random.randint(len(validloader.dataset))
        else:
            target_id = args.target_id
        ground_truth, labels = validloader.dataset[target_id]
        if args.label_flip:
            labels = (labels + 1) % len(trainloader.dataset.classes)
        ground_truth, labels = (
            ground_truth.unsqueeze(0).to(**setup),
            torch.as_tensor((labels,), device=setup["device"]),
        )
    img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])


    # Run reconstruction
    #rec. from gradient
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
    print(f"Full gradient norm is {full_norm:e}.")

    # Run reconstruction in different precision
    if args.dtype != "float":
        if args.dtype in ["double", "float64"]:
            setup["dtype"] = torch.double
        elif args.dtype in ["half", "float16"]:
            setup["dtype"] = torch.half
        else:
            raise ValueError(f"Unknown data type argument {args.dtype}.")
        print(f"Model and input parameter moved to {args.dtype}-precision.")
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
        ground_truth = ground_truth.to(**setup)
        input_gradient = [g.to(**setup) for g in input_gradient]
        model.to(**setup)
        model.eval()

    if args.optim == "ours":
        config = dict(
            signed=args.signed,
            boxed=args.boxed,
            cost_fn=args.cost_fn,
            indices="def",
            weights="equal",
            lr=0.1,
            optim=args.optimizer,
            restarts=args.restarts,
            max_iterations=24_000,
            total_variation=args.tv,
            init="randn",
            filter="none",
            lr_decay=True,
            scoring_choice="loss",
        )
        
    elif args.optim == "zhu":
        config = dict(
            signed=False,
            boxed=False,
            cost_fn="l2",
            indices="def",
            weights="equal",
            lr=1e-4,
            optim="LBFGS",
            restarts=args.restarts,
            max_iterations=300,
            total_variation=args.tv,
            init=args.init,
            filter="none",
            lr_decay=False,
            scoring_choice=args.scoring_choice,
        )

    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images)
    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=args.dryrun)

    

    # Compute stats
    test_mse = (output - ground_truth).pow(2).mean().item()
    feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1 / ds)

    # Save the resulting image
    if args.save_image and not args.dryrun:
        os.makedirs(args.image_path, exist_ok=True)
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        rec_filename = (
            f'{validloader.dataset.classes[labels][0]}_{"trained" if args.trained_model else ""}'
            f"{args.model}_{args.cost_fn}-{args.target_id}.png"
        )
        torchvision.utils.save_image(output_denormalized, os.path.join(args.image_path, rec_filename))

        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        gt_filename = f"{validloader.dataset.classes[labels][0]}_ground_truth-{args.target_id}.png"
        torchvision.utils.save_image(gt_denormalized, os.path.join(args.image_path, gt_filename))
    else:
        rec_filename = None
        gt_filename = None

    # Save to a table:
    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

    inversefed.utils.save_to_table(
        args.table_path,
        name=f"exp_{args.name}",
        dryrun=args.dryrun,
        model=args.model,
        dataset=args.dataset,
        trained=args.trained_model,
        accumulation=args.accumulation,
        restarts=args.restarts,
        OPTIM=args.optim,
        cost_fn=args.cost_fn,
        indices=args.indices,
        weights=args.weights,
        scoring=args.scoring_choice,
        init=args.init,
        tv=args.tv,
        rec_loss=stats["opt"],
        psnr=test_psnr,
        test_mse=test_mse,
        feat_mse=feat_mse,
        target_id=target_id,
        seed=model_seed,
        timing=str(datetime.timedelta(seconds=time.time() - start_time)),
        dtype=setup["dtype"],
        epochs=defs.epochs,
        val_acc=None,
        rec_img=rec_filename,
        gt_img=gt_filename,
    )

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print("---------------------------------------------------")
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
    print("-------------Job finished.-------------------------")
