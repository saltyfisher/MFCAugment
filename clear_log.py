import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

log_dir = './logs/GD'

for f in Path(log_dir).rglob('*'):
    # f = Path('/workspace/MFCAugment/logs/GD/lymphoma_resnet18_180epochs_450imsize_GD_itrs1-03-20-16-08-44/events.out.tfevents.1742458124.858e3b4bc67d')
    if f.is_dir():
        continue
    acc = EventAccumulator(str(f))
    acc.Reload()  # 加载所有事件数据

    # 获取所有记录的标签（tags）
    tags = acc.Tags()["scalars"]  # 可替换为 'images', 'histograms' 等
    # 统计每个tag的数据量
    if tags == []:
        shutil.rmtree(str(f.parent))
        continue
    for tag in tags:
        events = acc.Scalars(tag)  # 获取该tag下的所有事件
        if len(events) < 10:
            shutil.rmtree(str(f.parent))
            break