import os
import re
import time
import argparse
from torch.utils.tensorboard import SummaryWriter


def sync_log(log_path, tb_logdir, follow=True):
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)
    print(f'[ TensorBoard Sync ] Streaming metrics from {log_path} to {tb_logdir}')
    print(f'[ TensorBoard Sync ] Run `tensorboard --logdir logs/MobileRobotNav-v0/diffusion` to view live plots.')

    pattern = re.compile(r'(\d+):\s+([\d\.]+)\s+\|\s+a0_loss:\s+([\d\.]+)')
    seen_steps = set()

    def process_file(f):
        new_entries = 0
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                if step not in seen_steps:
                    loss = float(match.group(2))
                    a0_loss = float(match.group(3))
                    writer.add_scalar('loss/total_loss', loss, step)
                    writer.add_scalar('loss/a0_loss', a0_loss, step)
                    seen_steps.add(step)
                    new_entries += 1
        if new_entries > 0:
            writer.flush()
        return new_entries

    if not os.path.exists(log_path):
        print(f'Waiting for log file {log_path}...')
        while not os.path.exists(log_path):
            time.sleep(1)

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        process_file(f)
        if follow:
            while True:
                line = f.readline()
                if line:
                    match = pattern.search(line)
                    if match:
                        step = int(match.group(1))
                        if step not in seen_steps:
                            loss = float(match.group(2))
                            a0_loss = float(match.group(3))
                            writer.add_scalar('loss/total_loss', loss, step)
                            writer.add_scalar('loss/a0_loss', a0_loss, step)
                            writer.flush()
                            seen_steps.add(step)
                else:
                    time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, help='Path to task.log')
    parser.add_argument('--tb_dir', type=str, default='logs/MobileRobotNav-v0/diffusion/navigation_H32_T20/tb', help='Tensorboard log dir')
    parser.add_argument('--no_follow', action='store_true', help='Do not follow live updates')
    args = parser.parse_args()

    sync_log(args.log, args.tb_dir, follow=not args.no_follow)
