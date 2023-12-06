import os
import numpy as np

import utils
import align_dataset_test as align_dataset
from config import CONFIG

from evals.phase_classification import evaluate_phase_classification, compute_ap
from evals.kendalls_tau import evaluate_kendalls_tau
from evals.phase_progression import evaluate_phase_progression

from train import AlignNet
import torch
from torch.utils.tensorboard import SummaryWriter

import random
import argparse
import glob
from natsort import natsorted


def get_embeddings(model, data, labels_npy, args):
    embeddings = []
    labels = []
    frame_paths = []
    names = []

    device = f"cuda:{args.device}"

    for act_iter in iter(data):
        for i, seq_iter in enumerate(act_iter):
            seq_embs = []
            seq_fpaths = []
            original = 0
            for _, batch in enumerate(seq_iter):
                a_X, a_name, a_frames = batch
                a_X = a_X.to(device).unsqueeze(0)
                original = a_X.shape[1] // 2

                #                 a_X = a_X[:,:a_X.shape[1]//2,:,:,:]
                #                 original = a_X.shape[1]//2

                #                 if ((args.num_frames*2)-a_X.shape[1]) > 0:
                #                     b = a_X[:, -1].clone()
                #                     b = torch.stack([b]*((args.num_frames*2)-a_X.shape[1]),axis=1).to(device)
                #                     a_X = torch.concat([a_X,b], axis=1)

                b = a_X[:, -1].clone()
                try:
                    b = torch.stack(
                        [b] * ((args.num_frames * 2) - a_X.shape[1]), axis=1
                    ).to(device)
                except:
                    b = torch.from_numpy(np.array([])).float().to(device)
                a_X = torch.concat([a_X, b], axis=1)
                a_emb = model(a_X)[:, :original, :]

                if args.verbose:
                    print(f"Seq: {i}, ", a_emb.shape)

                seq_embs.append(a_emb.squeeze(0).detach().cpu().numpy())

                seq_fpaths.extend(a_frames)

            seq_embs = np.concatenate(seq_embs, axis=0)

            name = str(a_name).split("/")[-1]
            # name = name[:8] + '/' + name[8:10] + '/' + name[10:]
            lab = labels_npy[int(name)]["label"]
            end = min(seq_embs.shape[0], len(lab))
            lab = lab[:end]  # .T
            seq_embs = seq_embs[:end]
            print(seq_embs.shape, len(lab))
            embeddings.append(seq_embs[:end])
            frame_paths.append(seq_fpaths)
            names.append(a_name)
            labels.append(lab)

    return embeddings, names, labels


def main(ckpts, args):
    summary_dest = os.path.join(args.dest, "eval_logs")

    result_path = os.path.join(args.dest, "eval_results/all_actions")
    os.makedirs(summary_dest, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    result = dict()

    for ckpt in ckpts:
        writer = SummaryWriter(summary_dest, filename_suffix="eval_logs")

        ckpt_result = dict()

        # get ckpt-step from the ckpt name
        _, ckpt_step = ckpt.split(".")[0].split("_")[-2:]
        ckpt_step = int(ckpt_step.split("=")[1])
        DEST = os.path.join(args.dest, "eval_step_{}".format(ckpt_step))

        ckpt_epoch = ckpt.split(".")[0].split("epoc")[1].split("_")[0]
        ckpt_epoch = int(ckpt_epoch.split("=")[1])

        device = f"cuda:{args.device}"
        model = AlignNet.load_from_checkpoint(ckpt, map_location=device)
        model.to(device)
        model.eval()

        # grad off
        torch.set_grad_enabled(False)

        if args.num_frames:
            CONFIG.TRAIN.NUM_FRAMES = args.num_frames
            CONFIG.EVAL.NUM_FRAMES = args.num_frames

        CONFIG.update(model.hparams.config)

        print(model.hparams)
        if args.data_path:
            data_path = args.data_path
        else:
            data_path = CONFIG.DATA_PATH
        data_path = "/root/src/dataset/Penn_Action/all_divided_dataset/"

        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        # lab_train_path = os.path.join(data_path, 'labels', 'train')
        # lab_val_path = os.path.join(data_path, 'labels', 'val')
        print("args.model_path:", args.model_path)
        # lab_name = "_".join(args.model_path.split("/")[4].split("_")[:-1]) + "_val"
        # labels = np.load(f"/root/src/dataset/penn_action_labels/test/squats/{lab_name}.npy", allow_pickle=True).item()
        labels = np.load(
            "/root/src/dataset/penn_action_labels/all_data.npy", allow_pickle=True
        ).item()

        # create dataset
        _transforms = utils.get_transforms(augment=False)

        random.seed(0)
        train_data = align_dataset.AlignData(
            train_path,
            args.batch_size,
            CONFIG.DATA,
            transform=_transforms,
            flatten=False,
        )
        val_data = align_dataset.AlignData(
            val_path, args.batch_size, CONFIG.DATA, transform=_transforms, flatten=False
        )

        all_classifications = []
        all_kendalls_taus = []
        all_phase_progressions = []
        ap5, ap10, ap15 = 0, 0, 0
        for i_action in range(train_data.n_classes):
            train_data.set_action_seq(i_action)
            val_data.set_action_seq(i_action)

            train_act_name = train_data.get_action_name(i_action)
            val_act_name = val_data.get_action_name(i_action)
            print(f"train_act_name: {train_act_name}, val_act_name: {val_act_name}")
            assert train_act_name == val_act_name

            # PennActionのための追加
            if train_act_name != args.action_name and args.action_name is not None:
                continue
            # if args.verbose:
            #     print(f'Getting embeddings for {train_act_name}...')
            train_embs, train_names, train_labels = get_embeddings(
                model, train_data, labels, args
            )
            val_embs, val_names, val_labels = get_embeddings(
                model, val_data, labels, args
            )
            # train_embs, train_names, train_labels = val_embs, val_names, val_labels

            # # save embeddings
            os.makedirs(DEST, exist_ok=True)
            DEST_TRAIN = os.path.join(DEST, f"train_{train_act_name}_embs.npy")
            DEST_VAL = os.path.join(DEST, f"val_{val_act_name}_embs.npy")

            np.save(
                DEST_TRAIN,
                {"embs": train_embs, "names": train_names, "labels": train_labels},
            )
            np.save(
                DEST_VAL, {"embs": val_embs, "names": val_names, "labels": val_labels}
            )

            train_embeddings = np.load(DEST_TRAIN, allow_pickle=True).tolist()
            val_embeddings = np.load(DEST_VAL, allow_pickle=True).tolist()

            train_embs, train_labels, train_names = (
                train_embeddings["embs"],
                train_embeddings["labels"],
                train_embeddings["names"],
            )
            val_embs, val_labels, val_names = (
                val_embeddings["embs"],
                val_embeddings["labels"],
                val_embeddings["names"],
            )

            # Evaluating Classification
            train_acc, val_acc = evaluate_phase_classification(
                ckpt_step,
                train_embs,
                train_labels,
                val_embs,
                val_labels,
                act_name=train_act_name,
                CONFIG=CONFIG,
                writer=writer,
                verbose=args.verbose,
            )
            ap5, ap10, ap15 = compute_ap(val_embs, val_labels)

            all_classifications.append([train_acc, val_acc])

            # Evaluating Kendall's Tau
            train_tau, val_tau = evaluate_kendalls_tau(
                train_embs,
                val_embs,
                stride=args.stride,
                kt_dist=CONFIG.EVAL.KENDALLS_TAU_DISTANCE,
                visualize=False,
            )
            all_kendalls_taus.append([train_tau, val_tau])

            print(f"Kendal's Tau: Stride = {args.stride} \n")
            print(f"Train = {train_tau}\n")
            print(f"Val = {val_tau}\n")

            writer.add_scalar(
                f"kendalls_tau/train_{train_act_name}", train_tau, global_step=ckpt_step
            )
            writer.add_scalar(
                f"kendalls_tau/val_{val_act_name}", val_tau, global_step=ckpt_step
            )
            ckpt_result["action"] = train_act_name
            ckpt_result["epoch"] = ckpt_epoch

            ckpt_result["kendalls_tau/train"] = train_tau
            ckpt_result["kendalls_tau/val"] = val_tau

            # Evaluating Phase Progression
        #             _train_dict = {'embs': train_embs, 'labels': train_labels}
        #             _val_dict = {'embs': val_embs, 'labels': val_labels}
        #             train_phase_scores, val_phase_scores = evaluate_phase_progression(_train_dict, _val_dict, "_".join(lab_name.split('_')[:-1]),
        #                                                                                 ckpt_step, CONFIG, writer, verbose=args.verbose)

        #             all_phase_progressions.append([train_phase_scores[-1], val_phase_scores[-1]])

        train_classification, val_classification = np.mean(all_classifications, axis=0)
        train_kendalls_tau, val_kendalls_tau = np.mean(all_kendalls_taus, axis=0)
        # train_phase_prog, val_phase_prog = np.mean(all_phase_progressions, axis=0)

        writer.add_scalar("metrics/AP@5_val", ap5, global_step=ckpt_step)
        writer.add_scalar("metrics/AP@10_val", ap10, global_step=ckpt_step)
        writer.add_scalar("metrics/AP@15_val", ap15, global_step=ckpt_step)

        ckpt_result["metrics/AP@5_val"] = ap5
        ckpt_result["metrics/AP@10_val"] = ap10
        ckpt_result["metrics/AP@15_val"] = ap15

        writer.add_scalar(
            "metrics/all_classification_train",
            train_classification,
            global_step=ckpt_step,
        )
        writer.add_scalar(
            "metrics/all_classification_val", val_classification, global_step=ckpt_step
        )

        writer.add_scalar(
            "metrics/all_kendalls_tau_train", train_kendalls_tau, global_step=ckpt_step
        )
        writer.add_scalar(
            "metrics/all_kendalls_tau_val", val_kendalls_tau, global_step=ckpt_step
        )

        ckpt_result["metrics/all_classification_train"] = train_classification
        ckpt_result["metrics/all_classification_val"] = val_classification
        ckpt_result["metrics/all_kendalls_tau_train"] = train_kendalls_tau
        ckpt_result["metrics/all_kendalls_tau_val"] = val_kendalls_tau

        # writer.add_scalar('metrics/all_phase_progression_train', train_phase_prog, global_step=ckpt_step)
        # writer.add_scalar('metrics/all_phase_progression_val', val_phase_prog, global_step=ckpt_step)

        print("metrics/AP@5_val", ap5, f"global_step={ckpt_step}")
        print("metrics/AP@10_val", ap10, f"global_step={ckpt_step}")
        print("metrics/AP@15_val", ap15, f"global_step={ckpt_step}")

        print(
            "metrics/all_classification_train",
            train_classification,
            f"global_step={ckpt_step}",
        )
        print(
            "metrics/all_classification_val",
            val_classification,
            f"global_step={ckpt_step}",
        )

        print(
            "metrics/all_kendalls_tau_train",
            train_kendalls_tau,
            f"global_step={ckpt_step}",
        )
        print(
            "metrics/all_kendalls_tau_val", val_kendalls_tau, f"global_step={ckpt_step}"
        )

        # print('metrics/all_phase_progression_train', train_phase_prog, f"global_step={ckpt_step}")
        # print('metrics/all_phase_progression_val', val_phase_prog, f"global_step={ckpt_step}")

        writer.flush()

        writer.close()

        result[ckpt_step] = ckpt_result

    np.save(os.path.join(result_path, "result_{}.npy".format(args.action_name)), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--dest", type=str, default="./")

    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--visualize", dest="visualize", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="Cuda device to be used")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_frames", type=int, default=None, help="Path to dataset")
    parser.add_argument(
        "--action_name", type=str, default=None, help="Select action name"
    )

    args = parser.parse_args()

    if os.path.isdir(args.model_path):
        ckpts = natsorted(glob.glob(os.path.join(args.model_path, "*.ckpt")))
    else:
        ckpts = [args.model_path]

    ckpt_mul = args.device
    main(ckpts, args)
